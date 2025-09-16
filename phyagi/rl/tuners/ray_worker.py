# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from vllm import RequestOutput

from phyagi.datasets.rl.packing import PackedBatch
from phyagi.models.parallel_utils import maybe_apply_context_parallel_to_inputs
from phyagi.rl.distributed_layout import DistributedLayout
from phyagi.rl.models.actor import Actor
from phyagi.rl.models.reference import Reference
from phyagi.rl.ray_utils import get_ray_logger
from phyagi.rl.rollout.vllm_worker import VLLMWorker
from phyagi.rl.tuners.ray_worker_config import RayWorkerConfig
from phyagi.utils.checkpoint import CheckpointManager


class RayWorker:
    """Ray worker."""

    def __init__(
        self,
        config: RayWorkerConfig,
        checkpoint_manager: CheckpointManager,
        distributed_layout: DistributedLayout,
        tokenizer: Union[str, PreTrainedTokenizerBase],
        skip_process_group_init: bool = False,
        total_training_steps: Optional[int] = None,
    ) -> None:
        """Initialize the worker.

        Args:
            config: Ray worker configuration.
            checkpoint_manager: Checkpoint manager for saving/loading checkpoints.
            distributed_layout: Distributed layout for the worker.
            tokenizer: Tokenizer to use.
            skip_process_group_init: Whether to skip process group initialization.
            total_training_steps: Total number of training steps.

        """

        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.distributed_layout = distributed_layout
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer

        self.total_training_steps = total_training_steps or self.config.max_steps
        if self.total_training_steps is None:
            raise ValueError("`total_training_steps` or `config.max_steps` must be defined, but got None.")

        self.output_dir = Path(self.config.output_dir)
        self.max_context_length = self.config.rollout.prompt_length + self.config.rollout.response_length
        self.logger = get_ray_logger(__name__)

        if not torch.distributed.is_initialized() and not skip_process_group_init:
            torch.distributed.init_process_group(backend="cuda:nccl,cpu:gloo")
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            self.logger.info(f"Process group initialized with local rank {local_rank}.")

        self.actor_device_mesh = self.distributed_layout.init_device_mesh(context="actor")

    def _maybe_prepare_context_parallel_packed_batch(self, packed_batch: PackedBatch) -> PackedBatch:
        cp_mesh = self.actor_device_mesh["context_parallel"]
        if cp_mesh.size() <= 1:
            return packed_batch

        packed_batch_dict = packed_batch.to_dict()
        packed_batch_dict = maybe_apply_context_parallel_to_inputs(
            packed_batch_dict,
            context_parallel_world_size=cp_mesh.size(),
            context_parallel_rank=cp_mesh.get_local_rank(),
        )

        return PackedBatch.from_dict(packed_batch_dict)

    def configure_models(self, build_reference_model: bool = True, build_rollout: bool = True) -> None:
        """Configure the models initialization.

        Args:
            build_reference_model: Whether to build the reference model.
            build_rollout: Whether to build the rollout model.

        """

        self.logger.info("Initializing actor model...")
        self.actor = Actor(self.config.actor, self.actor_device_mesh, self.checkpoint_manager, self.logger)
        self.logger.info("Actor model initialized.")

        self.ref = None
        if build_reference_model:
            self.logger.info("Initializing reference model...")
            self.ref = Reference(self.config.actor, self.actor_device_mesh, self.checkpoint_manager, self.logger)
            self.logger.info("Reference model initialized.")

        self.rollout = None
        if build_rollout:
            self.logger.info("Initializing rollout model...")
            self.rollout = VLLMWorker.from_mixformer_sequential(
                self.output_dir / "initial_rollout",
                self.actor.model,
                self.tokenizer,
                self.config.rollout,
            )
            self.logger.info("Rollout model initialized.")

        self.actor.configure_parallelisms()
        self.actor.configure_optimizers(total_training_steps=self.total_training_steps)

    def save_actor_model(self, checkpoint_path: Path) -> None:
        """Save the actor model.

        Args:
            checkpoint_path: Path to save the checkpoint.

        """

        self.actor.save_checkpoint(
            checkpoint_path,
            overwrite=True,
            save_optimizer_states=True,
            save_lr_scheduler_states=True,
            save_config=True,
        )

    def load_actor_model(
        self, checkpoint_dir: Path, load_optimizer_states: bool = True, load_scheduler_states: bool = True
    ) -> None:
        """Load the actor model.

        Args:
            checkpoint_dir: Path to the checkpoint directory.

        """

        self.actor.load_checkpoint(
            checkpoint_dir,
            load_optimizer_states=load_optimizer_states,
            load_scheduler_states=load_scheduler_states,
        )

    def save_reference_model(self, checkpoint_dir: Path) -> None:
        """Save the reference model.

        Args:
            checkpoint_dir: Path to save the checkpoint.

        """

        if self.ref is not None:
            self.ref.save_checkpoint(
                checkpoint_dir,
                overwrite=True,
                save_optimizer_states=False,
                save_lr_scheduler_states=False,
                save_config=True,
            )

    def load_reference_model(self, checkpoint_dir: Path) -> None:
        """Load the reference model.

        Args:
            checkpoint_dir: Path to the checkpoint directory.

        """

        if self.ref is not None:
            self.ref.load_checkpoint(checkpoint_dir, load_optimizer_states=False, load_scheduler_states=False)

    def generate_completions(
        self,
        prompts: List[int],
        num_repetitions: int = 1,
        sync_weights: bool = True,
        batch_size: Optional[int] = None,
        use_tqdm: bool = False,
        **generation_kwargs,
    ) -> List[RequestOutput]:
        """Generate completions for the given prompts.

        Args:
            prompts: Input prompts to generate completions for.
            num_repetitions: Number of repetitions for each prompt.
            sync_weights: Whether to synchronize weights with the rollout model.
            batch_size: Batch size for generation.
            use_tqdm: Whether to use ``tqdm`` for progress bar.

        Returns:
            List of generated completions.

        """

        with self.rollout.on_gpu():
            if sync_weights:
                with self.actor.on_gpu(model_only=True):
                    self.logger.info("Synchronizing actor weights with rollout...")
                    self.rollout.sync_weights(self.actor.model)
                    self.logger.info("Synchronization done.")

            if batch_size:
                batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
                batches = tqdm(batches, desc="Generating completions") if use_tqdm else batches
                self.logger.info(f"Generating completions using {len(batches)} batches of {batch_size} prompts...")

                completions = [
                    self.rollout.generate_from_input_ids(batch, n=num_repetitions, **generation_kwargs)
                    for batch in batches
                ]
                completions = [completion for batch in completions for completion in batch]
            else:
                self.logger.info(f"Generating completions for {len(prompts)} prompts...")
                completions = self.rollout.generate_from_input_ids(
                    prompts, n=num_repetitions, use_tqdm=use_tqdm, **generation_kwargs
                )

            self.logger.info("Completions generated.")

        self.logger.info("vLLM is now asleep.")

        return completions

    def compute_loss(self) -> Any:
        """Compute the loss."""

        raise NotImplementedError("`RayWorker` must implement `compute_loss()`.")

    def update_actor_policy(self) -> Dict[str, float]:
        """Update the actor policy.

        Returns:
            Dictionary containing training metrics.

        """

        raise NotImplementedError("`RayWorker` must implement `update_actor_policy()`.")
