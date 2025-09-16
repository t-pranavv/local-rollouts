# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Any, Dict, Optional, Union

import torch
from lightning.pytorch import LightningModule
from torch.distributed.tensor.parallel import loss_parallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MIXFORMER_SEQUENTIAL_MODEL_TYPE,
)
from phyagi.models.mixformer_sequential.parallel_mixformer_sequential import (
    apply_ac_mixformer_sequential,
    apply_cp_mixformer_sequential,
    apply_fsdp_mixformer_sequential,
    apply_tp_mixformer_sequential,
)
from phyagi.models.parallel_utils import maybe_apply_context_parallel_to_inputs
from phyagi.optimizers.registry import get_lr_scheduler, get_optimizer
from phyagi.trainers.pl.pl_strategies import DataContextTensorParallelStrategy
from phyagi.trainers.pl.pl_training_args import (
    PlLightningModuleArguments,
    PlStrategyArguments,
)
from phyagi.utils.type_utils import nullcontext_kwargs, rgetattr


def _get_training_context(enable_loss_parallel: bool = False) -> AbstractContextManager:
    @contextmanager
    def _context():
        with ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(loss_parallel())

            yield

    return _context


class TrainingLightningModule(LightningModule):
    """PyTorch Lightning module for pre-training and fine-tuning (including SFT) models."""

    def __init__(
        self,
        model: torch.nn.Module,
        lm_args: PlLightningModuleArguments,
        strategy_args: PlStrategyArguments,
        optimizer: Optional[Optimizer],
        scheduler: Optional[LRScheduler],
    ) -> None:
        super().__init__()

        self.model = model
        self.model_config = getattr(self.model, "config", None)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self._model_type = rgetattr(self.model, "config.model_type", None)
        self._lm_args = lm_args
        self._strategy_args = strategy_args

        # Training context is null by default (and can accept kwargs)
        self._training_context = nullcontext_kwargs

    def configure_model(self) -> None:
        # If strategy is a 3D-parallelism (data, context, and tensor) and we have a MixFormerSequential,
        # apply the corresponding parallelism strategies
        is_dctp_strategy = self._strategy_args.type == DataContextTensorParallelStrategy.STRATEGY_TYPE
        is_mixformer_sequential = self._model_type == MIXFORMER_SEQUENTIAL_MODEL_TYPE

        if is_dctp_strategy and is_mixformer_sequential:
            # Check if the model configuration has the required attributes for parallelism
            tp_size = getattr(self.model_config, "tp_size", None)
            if tp_size != self._strategy_args.tensor_parallel_size:
                raise ValueError(
                    f"`model.tp_size` must be equal to `strategy.tensor_parallel_size`, but got {tp_size} and {self._strategy_args.tensor_parallel_size}."
                )

            cp_size = getattr(self.model_config, "cp_size", None)
            if cp_size != self._strategy_args.context_parallel_size:
                raise ValueError(
                    f"`model.cp_size` must be equal to `strategy.context_parallel_size`, but got {cp_size} and {self._strategy_args.context_parallel_size}."
                )

            # Parallelism meshes
            cp_mesh = self.device_mesh["context_parallel"]
            tp_mesh = self.device_mesh["tensor_parallel"]
            dp_mesh = self.device_mesh["data_context_parallel"]

            # Context Parallelism (CP)
            if cp_mesh.size() > 1:
                apply_cp_mixformer_sequential(self.model, cp_mesh)

            # Tensor Parallelism (TP)
            if tp_mesh.size() > 1:
                apply_tp_mixformer_sequential(
                    self.model,
                    tp_mesh,
                    enable_async=self._strategy_args.tp_async,
                    enable_sequence_parallel=self._strategy_args.tp_sequence_parallel,
                    enable_loss_parallel=self._strategy_args.tp_loss_parallel,
                )

            # Activation Checkpointing (AC)
            if self._strategy_args.activation_checkpointing:
                apply_ac_mixformer_sequential(self.model)

            # Data Parallelism (FSDP)
            if dp_mesh.size() > 1:
                apply_fsdp_mixformer_sequential(
                    self.model,
                    dp_mesh,
                    self.trainer.precision,
                    compile=self._strategy_args.fsdp_compile,
                    cpu_offload=self._strategy_args.fsdp_cpu_offload,
                )

            # Training context
            # Used to enable loss parallelism
            self._training_context = _get_training_context(enable_loss_parallel=self._strategy_args.tp_loss_parallel)

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, Dict[str, Union[LRScheduler, str]]]]:
        optimizer = self.optimizer
        if optimizer is None:
            # If `optimizer` was not supplied, check if it is defined in the configuration
            if self._lm_args.optimizer is None:
                raise ValueError("`optimizer` must be defined, but got None.")

            # If `optimizer` is defined in the configuration, check if `type` is defined
            optimizer_type = self._lm_args.optimizer.get("type", None)
            if optimizer_type is None:
                raise ValueError("`optimizer.type` must be defined, but got None.")

            # Since we might have different set of keys for different optimizers we handle them separately
            optimizer_params = self._lm_args.optimizer.get("params", {})
            if optimizer_type == "dion":
                optimizer_params["device_mesh"] = self.device_mesh

            optimizer = get_optimizer(self.model, optimizer_type, **optimizer_params)

        scheduler = self.scheduler
        if scheduler is None:
            # If `scheduler` was not supplied, ignore if it is not defined in the configuration
            if self._lm_args.scheduler is None:
                return {"optimizer": optimizer}

            # If `scheduler` is defined in the configuration, check if `type` is defined
            scheduler_type = self._lm_args.scheduler.get("type", None)
            if scheduler_type is None:
                raise ValueError("`scheduler.type` must be defined, but got None.")
            scheduler_params = self._lm_args.scheduler.get("params", {})

            scheduler = get_lr_scheduler(optimizer, scheduler_type, **scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self._lm_args.scheduler.get("interval", "step"),
                "frequency": self._lm_args.scheduler.get("frequency", 1),
                "name": self._lm_args.scheduler.get("name", None),
            },
        }

    def forward(self, **inputs) -> torch.Tensor:
        with self._training_context():
            return self.model(**inputs)

    def backward(self, *args, **kwargs) -> None:
        with self._training_context():
            super().backward(*args, **kwargs)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        cp_mesh = self.device_mesh["context_parallel"] if self.device_mesh is not None else None
        if cp_mesh is not None:
            batch = maybe_apply_context_parallel_to_inputs(
                batch, context_parallel_world_size=cp_mesh.size(), context_parallel_rank=cp_mesh.get_local_rank()
            )

        outputs = self(**batch)
        return {"loss": outputs.loss}

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        outputs = self(**batch)
        return {"loss": outputs.loss}
