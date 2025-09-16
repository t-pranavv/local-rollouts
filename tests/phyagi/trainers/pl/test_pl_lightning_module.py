# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch

from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MIXFORMER_SEQUENTIAL_MODEL_TYPE,
)
from phyagi.trainers.pl.pl_lightning_module import TrainingLightningModule
from phyagi.trainers.pl.pl_strategies import DataContextTensorParallelStrategy
from phyagi.trainers.pl.pl_training_args import (
    PlLightningModuleArguments,
    PlStrategyArguments,
)


@pytest.fixture
def mock_model():
    model = MagicMock()
    model_config = MagicMock()
    model_config.model_type = MIXFORMER_SEQUENTIAL_MODEL_TYPE
    model_config.tp_size = 1
    model_config.cp_size = 1
    model.config = model_config
    model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
    return model


@pytest.fixture
def mock_lm_args():
    return PlLightningModuleArguments(
        optimizer={"type": "adamw", "params": {"lr": 0.001}},
        scheduler={
            "type": "step",
            "params": {"step_size": 1, "gamma": 0.1},
            "interval": "epoch",
            "frequency": 1,
            "name": "test-scheduler",
        },
    )


@pytest.fixture
def mock_strategy_args():
    return PlStrategyArguments(
        type=DataContextTensorParallelStrategy.STRATEGY_TYPE,
        tp_async=False,
        tp_loss_parallel=False,
        activation_checkpointing=False,
        fsdp_compile=False,
        fsdp_cpu_offload=False,
    )


@patch("phyagi.trainers.pl.pl_lightning_module.apply_tp_mixformer_sequential")
@patch("phyagi.trainers.pl.pl_lightning_module.apply_ac_mixformer_sequential")
@patch("phyagi.trainers.pl.pl_lightning_module.apply_fsdp_mixformer_sequential")
def test_training_lightning_module(mock_fsdp, mock_ac, mock_tp, mock_model, mock_lm_args, mock_strategy_args):
    module = TrainingLightningModule(
        model=mock_model, lm_args=mock_lm_args, strategy_args=mock_strategy_args, optimizer=None, scheduler=None
    )

    fake_mesh = {
        "data_parallel": MagicMock(size=MagicMock(return_value=2)),
        "data_context_parallel": MagicMock(size=MagicMock(return_value=2)),
        "context_parallel": MagicMock(size=MagicMock(return_value=1)),
        "tensor_parallel": MagicMock(size=MagicMock(return_value=2)),
    }
    with patch.object(TrainingLightningModule, "device_mesh", new_callable=PropertyMock) as mock_mesh:
        mock_mesh.return_value = fake_mesh

        module.trainer = MagicMock()
        module.trainer.precision = "bf16"

        module.configure_model()
        mock_tp.assert_called_once()
        mock_ac.assert_not_called()
        mock_fsdp.assert_called_once()

    config = module.configure_optimizers()
    assert "optimizer" in config
    assert isinstance(config["optimizer"], torch.optim.Optimizer)
    assert "lr_scheduler" in config
    assert isinstance(config["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.LRScheduler)

    class DummyModel(torch.nn.Module):
        def forward(self, **inputs):
            out = MagicMock()
            out.loss = torch.tensor(2.0)
            return out

    module.model = DummyModel()
    batch = {"input_ids": torch.randint(0, 100, (2, 4))}

    train_output = module.training_step(batch, 0)
    assert train_output["loss"] == torch.tensor(2.0)

    val_output = module.validation_step(batch, 0)
    assert val_output["loss"] == torch.tensor(2.0)
