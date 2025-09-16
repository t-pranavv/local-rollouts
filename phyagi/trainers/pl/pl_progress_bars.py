# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Mapping, Optional, Union

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import _update_n, convert_inf
from typing_extensions import override


class TQDMStepProgressBar(TQDMProgressBar):
    """PyTorch Lightning custom progress bar.

    This progress bar is customized to display the number of training steps
    (with gradient accumulation) instead the number of training batches.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def total_train_steps(self) -> int:
        return int(self.trainer.num_training_batches / self.trainer.accumulate_grad_batches) + 1

    @property
    def total_train_batches(self) -> int:
        if self.trainer.max_epochs == -1 and self.trainer.max_steps is not None and self.trainer.max_steps > 0:
            remaining_steps = self.trainer.max_steps - self.trainer.global_step
            return min(self.total_train_steps, remaining_steps)
        return self.total_train_steps

    def _should_update(self, current: int, total: int) -> bool:
        return (
            self.is_enabled
            and current % self.trainer.accumulate_grad_batches == 0
            and (current % self.refresh_rate == 0 or current == total)
        )

    @override
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[Union[torch.Tensor, Mapping[str, Any]]],
        batch: Any,
        batch_idx: int,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, n // self.trainer.accumulate_grad_batches)
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
