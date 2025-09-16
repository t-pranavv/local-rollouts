# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import time
from typing import Any, Dict, Type

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from torch.optim import Optimizer

from phyagi.trainers.flops_utils import estimate_tflops, get_peak_tflops
from phyagi.utils.type_utils import rgetattr


class MetricLogCallback(Callback):
    """PyTorch Lightning callback to log metrics."""

    def __init__(self, log_every_n_steps: int = 50, enable_loss_parallel: bool = False) -> None:
        """Initialize the callback.

        Args:
            log_every_n_steps: Number of steps between logging metrics.
            enable_loss_parallel: When using loss parallelism, the loss is computed in parallel and can
                not be logged with `sync_dist=True`.

        """

        super().__init__()

        self._log_every_n_steps = log_every_n_steps
        self._enable_loss_parallel = enable_loss_parallel

        self._total_runtime = time.time()
        self._batch_start_time = 0.0
        self._batch_end_time = 0.0
        self._step_time = 0.0

        self._n_accumulated_batches = 0
        self._loss = 0.0
        self._peak_tflops = get_peak_tflops()

    def _calculate_throughput_metrics(self, trainer: Trainer, pl_module: LightningModule) -> Dict[str, float]:
        samples_per_second = trainer.train_batch_size / self._step_time
        tokens_per_second = samples_per_second * trainer.seq_len
        tflops = estimate_tflops(
            self._step_time,
            n_layer=rgetattr(pl_module, "model_config.n_layer", 1),
            n_embd=rgetattr(pl_module, "model_config.n_embd", 1),
            vocab_size=rgetattr(pl_module, "model_config.vocab_size", 1),
            seq_len=trainer.seq_len,
            batch_size=trainer.train_batch_size,
            activation_checkpointing=rgetattr(trainer, "args.strategy.activation_checkpointing", False),
        )

        return {
            "train/step_runtime": self._step_time,
            "train/samples_per_second": samples_per_second,
            "train/samples_per_second_per_gpu": samples_per_second / trainer.world_size,
            "train/tokens_per_second": tokens_per_second,
            "train/tokens_per_second_per_gpu": tokens_per_second / trainer.world_size,
            "train/tflops": tflops,
            "train/tflops_per_gpu": tflops / trainer.world_size,
            "train/mfu": tflops / trainer.world_size / self._peak_tflops,
        }

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        self._batch_start_time = time.time()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        self._batch_end_time = time.time()

        self._step_time += self._batch_end_time - self._batch_start_time
        self._n_accumulated_batches += 1
        self._loss += outputs["loss"]

        # When `n_accumulated_batches` reaches the number of accumulated batches,
        # we log the metrics and reset the counters
        if self._n_accumulated_batches == trainer.accumulate_grad_batches:
            if trainer.global_step % self._log_every_n_steps == 0:
                self.log_dict(
                    {
                        "train/loss": self._loss,
                        "train/ppl": math.exp(self._loss),
                    },
                    on_step=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=not self._enable_loss_parallel,
                )

                self.log_dict(
                    {
                        "train/total_runtime": time.time() - self._total_runtime,
                        "train/progress": trainer.global_step / trainer.max_steps,
                        "train/epoch": trainer.current_epoch,
                        "train/step": trainer.global_step,
                        "train/batch_size": trainer.train_batch_size,
                        "train/n_samples": trainer.train_batch_size * trainer.global_step,
                        "train/n_tokens": trainer.train_batch_size * trainer.global_step * trainer.seq_len,
                        **self._calculate_throughput_metrics(trainer, pl_module),
                    },
                    on_step=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )

            self._step_time = 0.0
            self._n_accumulated_batches = 0
            self._loss = 0.0

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.log_dict(
            {
                "eval/loss": outputs["loss"],
                "eval/ppl": torch.exp(outputs["loss"]),
            },
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=not self._enable_loss_parallel,
        )


class OptimizerLogCallback(Callback):
    """PyTorch Lightning callback to log optimizer-based information."""

    def __init__(self, log_every_n_steps: int = 50) -> None:
        """Initialize the callback.

        Args:
            log_every_n_steps: Number of steps between logging optimizer-based information.

        """

        super().__init__()

        self._log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer) -> None:
        def _calculate_global_norm(norm_type: float = 2.0) -> float:
            exp_norms = [
                p.grad.detach().data.norm(norm_type).item() ** norm_type
                for p in pl_module.parameters()
                if p.grad is not None
            ]
            if exp_norms:
                return sum(exp_norms) ** (1.0 / norm_type)
            return 0.0

        if trainer.global_step % self._log_every_n_steps == 0:
            self.log_dict(
                {
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/gradient_norm": _calculate_global_norm(),
                },
                on_step=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
