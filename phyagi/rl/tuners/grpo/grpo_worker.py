# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizerBase

from phyagi.datasets.rl.packing import PackedBatch
from phyagi.models.parallel_utils import clip_grad_by_total_norm_, get_grad_norm
from phyagi.rl.distributed_layout import DistributedLayout
from phyagi.rl.tuners.grpo.grpo_config import RayGRPOConfig
from phyagi.rl.tuners.ray_worker import RayWorker
from phyagi.utils.checkpoint import CheckpointManager


class RayGRPOWorker(RayWorker):
    """Ray GRPO worker."""

    def __init__(
        self,
        config: RayGRPOConfig,
        checkpoint_manager: CheckpointManager,
        distributed_layout: DistributedLayout,
        tokenizer: Union[str, PreTrainedTokenizerBase],
        skip_process_group_init: bool = False,
        total_training_steps: Optional[int] = None,
    ) -> None:
        """Initialize the worker.

        Args:
            config: GRPO configuration.
            checkpoint_manager: Checkpoint manager for saving/loading checkpoints.
            distributed_layout: Distributed layout for the worker.
            tokenizer: Tokenizer to use.
            skip_process_group_init: Whether to skip process group initialization.
            total_training_steps: Total number of training steps.

        """

        super().__init__(
            config,
            checkpoint_manager,
            distributed_layout,
            tokenizer,
            skip_process_group_init=skip_process_group_init,
            total_training_steps=total_training_steps,
        )

    def _get_loss_scale(self, packed_batch: PackedBatch, loss_normalization: str) -> Union[None, int, torch.Tensor]:
        if loss_normalization == "sequence":
            seqlens = packed_batch.cu_seqlens.diff()
            valid_seqlens = packed_batch.valid_seqlens
            masks = packed_batch.masks

            return valid_seqlens.repeat_interleave(seqlens).reshape_as(masks) + 1e-3

        if loss_normalization == "max_context":
            return self.max_context_length

        return None

    def compute_loss(
        self,
        per_token_logps: torch.FloatTensor,
        completion_mask: torch.BoolTensor,
        ref_per_token_logps: Optional[torch.FloatTensor],
        old_per_token_logps: Optional[torch.FloatTensor],
        advantages: torch.FloatTensor,
        entropy: Optional[torch.FloatTensor] = None,
        loss_scale: Union[float, torch.Tensor, None] = None,
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """Compute the GRPO loss.

        Args:
            per_token_logps: Per-token log probabilities.
            completion_mask: Completion mask.
            ref_per_token_logps: Reference per-token log probabilities.
            old_per_token_logps: Old per-token log probabilities.
            advantages: Advantages.
            entropy: Optional average entropy for the batch.
            loss_scale: Loss scaling factor.
                If ``float``, it is scaled by this ``batch_size``.
                If ``None``, it is scaled by the number of valid tokens in the packed batch.
                If a tensor of the same shape as ``completion_mask``, it is scaled by the corresponding value.

        Returns:
            Tuple with loss and dictionary of metrics (mean KL divergence, clip ratio, and entropy).

        """

        device = per_token_logps.device
        old_per_token_logps = (
            old_per_token_logps.to(device) if old_per_token_logps is not None else per_token_logps.detach()
        )

        epsilon_high = self.config.epsilon_high or self.config.epsilon_low
        epsilon_low = self.config.epsilon_low or self.config.epsilon_high

        # Shift the masks and advantages to match the per-token log probabilities
        # (that are computed for the next token)
        shifted_mask = completion_mask[:, 1:].to(device)
        shifted_advantages = advantages[:, 1:].to(device)

        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
        per_token_loss1 = coef_1 * shifted_advantages
        per_token_loss2 = coef_2 * shifted_advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        mean_kl = None
        num_valid_tokens = shifted_mask.sum() + 1e-3

        if self.config.kl_coeff != 0:
            # Reference: https://github.com/volcengine/verl/blob/3f63715a96ac8831d3624b8584d2aba1afc9c3fa/verl/trainer/ppo/core_algos.py#L1057
            ref_per_token_logps = ref_per_token_logps.to(device)
            kl = torch.clamp(ref_per_token_logps - per_token_logps, min=-20, max=20)
            per_token_kl = torch.clamp(torch.exp(kl) - kl - 1, min=-10, max=10)
            per_token_loss = per_token_loss + self.config.kl_coeff * per_token_kl
            mean_kl = (per_token_kl * shifted_mask).sum() / num_valid_tokens

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * shifted_mask).sum() / num_valid_tokens

        if isinstance(loss_scale, torch.Tensor):
            loss = ((per_token_loss * shifted_mask) / loss_scale[:, 1:].to(device)).sum()
        else:
            loss_scale = loss_scale or num_valid_tokens
            loss = (per_token_loss * shifted_mask).sum() / loss_scale

        if entropy is not None and self.config.entropy_coeff > 0:
            loss = loss - entropy * self.config.entropy_coeff

        return loss, {"train/mean_kl": mean_kl, "train/entropy": entropy, "train/clip_ratio": clip_ratio}

    def update_actor_policy(self, packed_batches: List[PackedBatch], temperature: float = 1.0) -> Dict[str, float]:
        """Update the actor policy.

        Args:
            packed_batches: List of packed batches.
            temperature: Temperature for the actor's log probabilities.

        Returns:
            Training loss, learning rate, mean KL divergence, clip ratio, and gradient norm.

        """

        self.logger.info(f"Updating actor policy with {len(packed_batches)} batches...")
        self.logger.info(f"Shapes: {[(b.tokens.shape, b.cu_seqlens) for b in packed_batches]}")

        old_per_token_logps, ref_outputs = None, {}
        metrics = dict.fromkeys(["train/mean_kl", "train/clip_ratio", "train/grad_norm", "train/entropy"], 0.0)

        with self.actor.on_gpu():
            device = self.actor.model.device
            total_loss = 0.0

            # `set_requires_gradient_sync=True` is required for memory-intensive workloads with FSDP
            # https://huggingface.co/docs/accelerate/concept_guides/gradient_synchronization#nosync-requires-additional-gpu-memory-when-using-fsdp
            self.actor.optimizer.zero_grad()
            self.actor.model.set_requires_gradient_sync(True)

            for _ in range(self.config.num_policy_updates_per_batch):
                for packed_batch in packed_batches:
                    packed_batch = self._maybe_prepare_context_parallel_packed_batch(packed_batch)
                    packed_batch.to(device)

                    loss_scale = self._get_loss_scale(packed_batch, self.config.loss_normalization)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        actor_outputs = self.actor.compute_logprobs(
                            packed_batch.tokens,
                            packed_batch.masks,
                            position_ids=packed_batch.position_ids,
                            cu_seqlens=packed_batch.cu_seqlens,
                            temperature=temperature,
                            compute_entropy=(self.config.entropy_coeff > 0),
                        )

                        if self.config.kl_coeff > 0.0:
                            with torch.no_grad():
                                ref_outputs = self.ref.compute_logprobs(
                                    packed_batch.tokens,
                                    packed_batch.masks,
                                    position_ids=packed_batch.position_ids,
                                    cu_seqlens=packed_batch.cu_seqlens,
                                    temperature=temperature,
                                )

                        step_loss, step_metrics = self.compute_loss(
                            actor_outputs["logprobs"],
                            packed_batch.masks,
                            ref_outputs.get("logprobs", None),
                            old_per_token_logps,
                            advantages=packed_batch.advantages,
                            entropy=actor_outputs["entropy"],
                            loss_scale=loss_scale,
                        )

                        # We multiply the loss by the number of data-parallel workers because FSDP will average the gradients
                        # across all workers, and we want to keep the loss scale consistent with the original batch
                        step_loss *= self.distributed_layout.actor_dp_size / (
                            self.config.train_batch_size * self.config.group_size
                        )

                    step_loss.float().backward()
                    total_loss += step_loss.item()

                    for key, value in step_metrics.items():
                        if key in metrics:
                            metrics[key] += value.item() if value is not None else 0.0

                with torch.no_grad():
                    grad_norm = get_grad_norm(
                        self.actor.model.parameters(),
                        dp_mesh=self.actor.device_mesh["data_context_parallel"],
                        tp_mesh=self.actor.device_mesh["tensor_parallel"],
                    )
                    clip_grad_by_total_norm_(
                        self.actor.model.parameters(),
                        max_norm=self.actor.config.gradient_clipping,
                        total_norm=grad_norm,
                    )

                # Normalize and update metrics with the gradient norm
                metrics = {
                    **{k: v / len(packed_batches) for k, v in metrics.items()},
                    "train/grad_norm": grad_norm,
                }

                # `step()` is only called after accumulating every batch gradients
                self.actor.optimizer.step()
                self.actor.lr_scheduler.step()

        torch.cuda.empty_cache()

        return {
            "train/loss": total_loss / len(packed_batches),
            "train/lr": self.actor.lr_scheduler.get_last_lr()[0],
            **metrics,
        }
