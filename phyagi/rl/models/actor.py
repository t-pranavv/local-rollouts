# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from logging import Logger, getLogger
from pathlib import Path
from typing import Dict, Generator, Optional

import torch
from torch.distributed import DeviceMesh
from torch.optim import Optimizer

from phyagi.models.mixformer_sequential.blocks.heads.logits_utils import (
    entropy_from_logits,
    logprobs_from_logits,
    logprobs_from_parallel_logits,
)
from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MIXFORMER_SEQUENTIAL_MODEL_TYPE,
)
from phyagi.models.mixformer_sequential.modeling_mixformer_sequential import (
    MixFormerSequentialForCausalLM,
)
from phyagi.models.mixformer_sequential.parallel_mixformer_sequential import (
    apply_ac_mixformer_sequential,
    apply_cp_mixformer_sequential,
    apply_fsdp_mixformer_sequential,
    apply_tp_mixformer_sequential,
)
from phyagi.models.registry import get_model
from phyagi.optimizers.schedulers.warmup_decay import WarmupDecayCooldownLR
from phyagi.rl.models.actor_config import ActorConfig
from phyagi.utils.checkpoint import CheckpointManager
from phyagi.utils.file_utils import save_json_file
from phyagi.utils.import_utils import is_torchao_available

AdamW8bit = None
if is_torchao_available():
    from torchao.optim.adam import AdamW8bit


@torch.no_grad()
def _move_optimizer_state(optimizer: Optimizer, device: str) -> None:
    if not optimizer.state:
        return

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device, non_blocking=False)


class Actor:
    """Ray actor model (with FSDP support)."""

    def __init__(
        self,
        config: ActorConfig,
        device_mesh: DeviceMesh,
        checkpoint_manager: Optional[CheckpointManager] = None,
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialize the actor model.

        Args:
            config: Actor model configuration.
            device_mesh: Device mesh.
            checkpoint_manager: Checkpoint manager for saving/loading checkpoints.
            logger: Logger for the actor model.

        """

        self.config = deepcopy(config)
        self.device_mesh = device_mesh
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.logger = logger or getLogger(__name__)

        self.manual_offload = self.config.manual_offload
        self.fsdp_offload = self.config.fsdp_offload
        self.precision = self.config.dtype

        # TP requires a full state dict to avoid memory pressure when using asynchronous checkpointing
        self.full_state_dict = self.checkpoint_manager.mode == "async" and self.config.tensor_parallel_size > 1

        self.optimizer = None
        self.lr_scheduler = None

        # Need to build the model with `fp32` to use full precision in the optimizer states and gradients when using fsdp
        # https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed#on-differences-in-data-precision-handling
        self.model = self.configure_model(**self.config.model, precision="float32")

        # Update the model configuration with the full configuration (with defaults and user overrides)
        # and set the precision to the one specified in the config
        self.config.model = self.model.config.to_diff_dict()
        self.config.model["torch_dtype"] = self.precision

        # Compile `entropy_from_logits` for better performance and memory efficiency
        self._entropy_from_logits = torch.compile(entropy_from_logits, dynamic=True)

    @staticmethod
    def configure_model(
        pretrained_model_name_or_path: Optional[str] = None, precision: Optional[str] = None, **kwargs
    ) -> MixFormerSequentialForCausalLM:
        """Configure the model for the actor.

        Args:
            pretrained_model_name_or_path: Pretrained model name or path.
            precision: Precision to use for the model.

        Returns:
            Configured model.

        """

        kwargs["model_type"] = MIXFORMER_SEQUENTIAL_MODEL_TYPE
        kwargs["torch_dtype"] = precision or kwargs.get("torch_dtype")

        pretrained_model_name_or_path = (
            pretrained_model_name_or_path
            or kwargs.pop("pretrained_model_name_or_path", None)
            or kwargs.get("_name_or_path", None)
        )

        model = get_model(pretrained_model_name_or_path, **kwargs)

        return model

    def configure_optimizers(self, total_training_steps: int) -> None:
        """Configure optimizers and learning rate scheduler for the actor model.

        Args:
            total_training_steps: Total number of training steps.

        """

        if self.config.adam_8bit:
            if AdamW8bit is None:
                raise ImportError("`torchao` is not available. Install `torchao` to use 8-bit AdamW optimizer.")
            self.optimizer = AdamW8bit(self.model.parameters(), **self.config.optimizer)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **self.config.optimizer)

        self.lr_scheduler = WarmupDecayCooldownLR(
            self.optimizer, total_num_steps=total_training_steps, **self.config.scheduler
        )

    def configure_parallelisms(self) -> None:
        """Configure parallelisms for the actor model."""

        cp_mesh = self.device_mesh["context_parallel"]
        tp_mesh = self.device_mesh["tensor_parallel"]
        dp_mesh = self.device_mesh["data_context_parallel"]

        if cp_mesh.size() > 1:
            apply_cp_mixformer_sequential(self.model, cp_mesh, varlen=True)
        if tp_mesh.size() > 1:
            apply_tp_mixformer_sequential(self.model, tp_mesh, enable_loss_parallel=self.config.tp_loss_parallel)
        if self.config.activation_checkpointing:
            apply_ac_mixformer_sequential(self.model)
        apply_fsdp_mixformer_sequential(self.model, dp_mesh, self.precision, cpu_offload=self.fsdp_offload)

        if self.manual_offload:
            self.model.to("cpu", non_blocking=False)

        # Either `fsdp_offload` or `manual_offload` leaves allocated GPU memory, so we need to empty the cache
        torch.cuda.empty_cache()

    @contextmanager
    @torch.no_grad()
    def on_gpu(
        self, model_only: bool = False, optimizer_only: bool = False, empty_cache: bool = True
    ) -> Generator[None, None, None]:
        """Context manager to move the model and optimizer to GPU.

        Args:
            model_only: Whether to move only the model to GPU.
            optimizer_only: Whether to move only the optimizer to GPU.
            empty_cache: Whether to empty the CUDA cache after offloading.

        Yields:
            Context manager that moves the model and optimizer to GPU.

        """

        if not self.manual_offload:
            yield

        else:
            if model_only and optimizer_only:
                raise ValueError("`model_only` and `optimizer_only` cannot be used together.")

            try:
                device = torch.cuda.current_device()
                if not optimizer_only:
                    self.model.to(device, non_blocking=True)
                if not model_only:
                    _move_optimizer_state(self.optimizer, device)

                yield

            finally:
                if not optimizer_only:
                    self.model.to("cpu", non_blocking=True)
                if not model_only:
                    _move_optimizer_state(self.optimizer, "cpu")

        if empty_cache:
            torch.cuda.empty_cache()

    def compute_logprobs(
        self,
        input_ids: torch.Tensor,
        assistant_masks: torch.BoolTensor,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.IntTensor] = None,
        temperature: Optional[float] = 1.0,
        compute_entropy: bool = False,
    ) -> Dict[str, torch.FloatTensor]:
        """Compute log probabilities for the given input identifiers and assistant masks.

        Args:
            input_ids: Input identifiers tensor.
            assistant_masks: Assistant masks tensor.
            position_ids: Position identifiers tensor
            cu_seqlens: Cumulative sequence lengths tensor.
            temperature: Temperature for scaling logits.
            compute_entropy: Whether to compute entropy.

        Returns:
            Dictionary with log probablities and entropy (optional).

        """

        logits = self.model(input_ids, position_ids=position_ids, cu_seqlens=cu_seqlens, use_cache=False).logits
        logits = logits / temperature

        labels = torch.where(assistant_masks, input_ids, -100).to(logits.device)

        return {
            "logprobs": (
                logprobs_from_parallel_logits(logits, labels)
                if self.config.tensor_parallel_size > 1 and self.config.tp_loss_parallel
                else logprobs_from_logits(logits, labels)
            ),
            "entropy": self._entropy_from_logits(logits, assistant_masks) if compute_entropy else None,
        }

    def save_checkpoint(
        self,
        checkpoint_path: Path,
        overwrite: bool = True,
        save_optimizer_states: bool = True,
        save_lr_scheduler_states: bool = True,
        save_config: bool = True,
    ) -> None:
        """Save a checkpoint.

        Args:
            checkpoint_path: Path to save the checkpoint.
            overwrite: Whether to overwrite the existing checkpoint.
            save_optimizer_states: Whether to save the optimizer states.
            save_lr_scheduler_states: Whether to save the learning rate scheduler states.
            save_config: Whether to save the model configuration.

        """

        self.logger.info(f"Saving checkpoint: {checkpoint_path}")

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            if save_config:
                save_json_file(self.config.to_dict(), checkpoint_path / "actor_config.json")

        self.checkpoint_manager.save(
            checkpoint_path,
            self.model,
            optimizer=self.optimizer if save_optimizer_states else None,
            scheduler=self.lr_scheduler if save_lr_scheduler_states else None,
            overwrite=overwrite,
            full_state_dict=self.full_state_dict,
            cpu_offload=True,
        )

        self.logger.info("Checkpoint saved (or scheduled to be saved).")

    def load_checkpoint(
        self, load_dir: Path, load_optimizer_states: bool = True, load_scheduler_states: bool = True
    ) -> None:
        """Load a checkpoint.

        Args:
            load_dir: Directory to load the checkpoint from.
            load_optimizer_states: Whether to load the optimizer states.
            load_scheduler_states: Whether to load the learning rate scheduler states.

        """

        self.logger.info(f"Loading checkpoint: {load_dir}")

        if not load_dir.exists():
            raise FileNotFoundError(f"'{load_dir}' is not a valid path.")

        self.checkpoint_manager.load(
            load_dir,
            self.model,
            optimizer=self.optimizer if load_optimizer_states else None,
            scheduler=self.lr_scheduler if load_scheduler_states else None,
            full_state_dict=self.full_state_dict,
            cpu_offload=True,
        )

        self.logger.info("Checkpoint loaded.")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Path,
        device_mesh: DeviceMesh,
        config: Optional[ActorConfig] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        logger: Optional[Logger] = None,
    ) -> Actor:
        """Create a Ray actor from a checkpoint.

        Args:
            checkpoint_dir: Directory to load the checkpoint from.
            config: Actor model configuration.
            device_mesh: Device mesh.
            checkpoint_manager: Checkpoint manager for saving/loading checkpoints.
            logger: Logger for the actor model.
            load_optimizer_states: Whether to load the optimizer states.
            load_scheduler_states: Whether to load the learning rate scheduler states.

        Returns:
            Ray actor instance.

        """
        checkpoint_dir = Path(checkpoint_dir)

        if config is None:
            config_file = checkpoint_dir / "actor_config.json"
            assert config_file.exists(), f"Config file '{config_file}' does not exist."

            config = ActorConfig(**json.load(config_file.open("r")))

        actor = cls(config, device_mesh, checkpoint_manager, logger)
        actor.load_checkpoint(checkpoint_dir, load_optimizer_states=False, load_scheduler_states=False)

        return actor
