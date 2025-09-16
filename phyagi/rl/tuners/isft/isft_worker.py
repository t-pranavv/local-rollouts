# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase

from phyagi.datasets.rl.packing import PackedBatch
from phyagi.models.parallel_utils import clip_grad_by_total_norm_, get_grad_norm
from phyagi.rl.distributed_layout import DistributedLayout
from phyagi.rl.tuners.isft.isft_config import RayISFTConfig
from phyagi.rl.tuners.ray_worker import RayWorker
from phyagi.utils.checkpoint import CheckpointManager


class RayISFTWorker(RayWorker):
    """Ray ISFT worker."""

    def __init__(
        self,
        config: RayISFTConfig,
        checkpoint_manager: CheckpointManager,
        distributed_layout: DistributedLayout,
        tokenizer: Union[str, PreTrainedTokenizerBase],
        skip_process_group_init: bool = False,
        total_training_steps: Optional[int] = None,
    ) -> None:
        """Initialize the worker.

        Args:
            config: ISFT configuration.
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

    def configure_models(self) -> None:
        return super().configure_models(build_reference_model=False)

    def compute_loss(
        self,
        per_token_logps: torch.FloatTensor,
        completion_mask: torch.BoolTensor,
        advantages: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute the loss.

        Args:
            per_token_logps: Per-token log probabilities.
            completion_mask: Completion mask.
            advantages: Advantages.

        Returns:
            Loss value.

        """

        # Shift the masks and advantages to match the per-token log probabilities
        # (that are computed for the next token)
        shifted_mask = completion_mask[:, 1:]
        shifted_advantages = advantages[:, 1:]

        per_token_loss = -per_token_logps * shifted_advantages
        loss = (per_token_loss * shifted_mask).sum() / (shifted_mask.sum() + 1e-3)

        return loss

    def update_actor_policy(
        self,
        packed_batches: List[PackedBatch],
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """Update the actor policy.

        Args:
            packed_batches: List of packed batches.
            temperature: Temperature for sampling.

        Returns:
            Training loss, learning rate, and gradient norm.

        """

        self.logger.info(f"Updating actor policy with {len(packed_batches)} batches...")
        self.logger.info(f"Shapes: {[b.tokens.shape for b in packed_batches]}")

        with self.actor.on_gpu():
            device = self.actor.model.device
            total_loss = 0.0

            # `set_requires_gradient_sync=True` is required for memory-intensive workloads with FSDP
            # https://huggingface.co/docs/accelerate/concept_guides/gradient_synchronization#nosync-requires-additional-gpu-memory-when-using-fsdp
            self.actor.optimizer.zero_grad()
            self.actor.model.set_requires_gradient_sync(True)

            for packed_batch in packed_batches:
                packed_batch = self._maybe_prepare_context_parallel_packed_batch(packed_batch)
                packed_batch.to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    actor_outputs = self.actor.compute_logprobs(
                        packed_batch.tokens,
                        packed_batch.masks,
                        position_ids=packed_batch.position_ids,
                        cu_seqlens=packed_batch.cu_seqlens,
                        temperature=temperature,
                    )

                    step_loss = self.compute_loss(
                        actor_outputs["logprobs"], packed_batch.masks, advantages=packed_batch.advantages
                    )

                step_loss.float().backward()
                total_loss += step_loss.item()

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

            # `step()` is only called after accumulating every batch gradients
            self.actor.optimizer.step()
            self.actor.lr_scheduler.step()

        torch.cuda.empty_cache()

        return {
            "train/loss": total_loss / len(packed_batches),
            "train/lr": self.actor.lr_scheduler.get_last_lr()[0],
            "train/grad_norm": grad_norm,
        }
