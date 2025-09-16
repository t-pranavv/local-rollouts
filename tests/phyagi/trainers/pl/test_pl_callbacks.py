# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import time
from unittest.mock import MagicMock

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer

from phyagi.trainers.pl.pl_callbacks import MetricLogCallback, OptimizerLogCallback


@pytest.fixture
def mock_trainer():
    trainer = MagicMock(spec=Trainer)
    trainer.args = MagicMock()
    trainer.global_step = 10
    trainer.max_steps = 100
    trainer.current_epoch = 1
    trainer.train_batch_size = 4
    trainer.seq_len = 128
    trainer.world_size = 1
    trainer.accumulate_grad_batches = 2
    return trainer


@pytest.fixture
def mock_model():
    model = MagicMock(spec=LightningModule)
    model.model = MagicMock()
    model.model.config = MagicMock()
    return model


@pytest.fixture
def mock_optimizer():
    optimizer = MagicMock()
    optimizer.param_groups = [{"lr": 0.001}]
    return optimizer


@pytest.fixture
def mock_batch():
    return {"input_ids": torch.tensor([[0, 1, 2, 3]])}


@pytest.fixture
def mock_outputs():
    return {"loss": torch.tensor(1.0)}


def test_metric_log_callback_logging(mock_trainer, mock_model, mock_batch, mock_outputs):
    callback = MetricLogCallback(log_every_n_steps=10)
    callback.log_dict = MagicMock()
    callback._total_runtime = time.time()

    callback._n_accumulated_batches = mock_trainer.accumulate_grad_batches - 1
    callback._loss = math.log(10)
    callback._step_time = 0.1
    mock_trainer.global_step = 10

    callback.on_train_batch_end(mock_trainer, mock_model, mock_outputs, mock_batch, batch_idx=0)
    assert callback.log_dict.call_count == 2

    calls = callback.log_dict.call_args_list
    first_call_args = calls[0][0][0]
    assert "train/loss" in first_call_args
    assert "train/ppl" in first_call_args


def test_metric_log_callback_on_validation_batch_end(mock_trainer, mock_model, mock_batch, mock_outputs):
    callback = MetricLogCallback()
    callback.log_dict = MagicMock()

    callback.on_validation_batch_end(mock_trainer, mock_model, mock_outputs, mock_batch, batch_idx=0)

    callback.log_dict.assert_called_once()
    call_args = callback.log_dict.call_args[0][0]
    assert "eval/loss" in call_args
    assert "eval/ppl" in call_args


def test_optimizer_log_callback(mock_trainer, mock_model, mock_optimizer):
    callback = OptimizerLogCallback(log_every_n_steps=5)
    callback.log_dict = MagicMock()

    param = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))
    param.grad = torch.tensor([0.1, 0.2, 0.3])
    mock_model.parameters.return_value = [param]

    mock_trainer.global_step = 10
    callback.on_before_optimizer_step(mock_trainer, mock_model, mock_optimizer)

    callback.log_dict.assert_called_once()
    call_args = callback.log_dict.call_args[0][0]
    assert "train/learning_rate" in call_args
    assert "train/gradient_norm" in call_args
