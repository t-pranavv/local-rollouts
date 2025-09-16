# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.progress.tqdm_progress import convert_inf

from phyagi.trainers.pl.pl_progress_bars import TQDMStepProgressBar


def test_tqdm_step_progress_bar():
    trainer = MagicMock(spec=Trainer)
    trainer.accumulate_grad_batches = 2
    trainer.current_epoch = 0
    trainer.total_train_batches = 10

    progress_bar = TQDMStepProgressBar()
    pl_module = MagicMock()
    batch = MagicMock()

    progress_bar._trainer = trainer
    progress_bar.train_progress_bar = MagicMock()
    progress_bar.train_progress_bar.set_postfix = MagicMock()
    progress_bar.train_progress_bar.set_description = MagicMock()
    progress_bar.train_progress_bar.reset = MagicMock()
    progress_bar.train_progress_bar.total = 10
    assert progress_bar.total_train_steps == 2

    progress_bar.on_train_epoch_start(trainer, None)
    progress_bar.train_progress_bar.reset.assert_called_once_with(convert_inf(progress_bar.total_train_steps))
    progress_bar.train_progress_bar.set_description.assert_called_once_with(f"Epoch {trainer.current_epoch}")
    progress_bar.on_train_batch_end(trainer, pl_module, None, batch, 3)
    progress_bar.train_progress_bar.set_postfix.assert_called_once()

    assert progress_bar._should_update(2, 10)
    assert not progress_bar._should_update(1, 10)
