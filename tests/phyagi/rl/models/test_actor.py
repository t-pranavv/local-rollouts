# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock, patch

import pytest
import torch

from phyagi.rl.models.actor import Actor


@pytest.fixture
def dummy_config():
    mock_config = MagicMock()
    mock_config.fsdp_offload = False
    mock_config.manual_offload = False
    mock_config.dtype = "float16"
    mock_config.use_meta_tensor = False
    mock_config.adam_8bit = False
    mock_config.checkpoint_mode = "sync"
    mock_config.optimizer = {"lr": 1e-3}
    mock_config.scheduler = {"warmup_steps": 10, "cooldown_steps": 5}
    mock_config.activation_checkpointing = False
    mock_config.model = {"pretrained_model_name_or_path": "dummy"}
    return mock_config


@pytest.fixture
def dummy_checkpoint_manager():
    checkpoint_manager = MagicMock()
    checkpoint_manager.mode = "sync"
    checkpoint_manager.save = MagicMock()
    checkpoint_manager.load = MagicMock()
    return checkpoint_manager


@pytest.fixture
def dummy_logger():
    return MagicMock()


@pytest.fixture
def dummy_mesh():
    mesh = MagicMock()
    mesh.get_group.return_value = "mock_group"
    return mesh


@patch("phyagi.rl.models.actor.get_model")
def test_actor(mock_get_model, dummy_config, dummy_mesh, dummy_checkpoint_manager, dummy_logger):
    mock_model = MagicMock()
    mock_model.config.to_diff_dict.return_value = {"mock_key": "mock_value"}
    mock_get_model.return_value = mock_model

    actor = Actor(dummy_config, dummy_mesh, dummy_checkpoint_manager, dummy_logger)

    assert actor.model is mock_model
    assert actor.optimizer is None
    assert actor.lr_scheduler is None
    assert actor.config.model["mock_key"] == "mock_value"
    assert actor.config.model["torch_dtype"] == "float16"


@patch("phyagi.rl.models.actor.get_model")
@patch("phyagi.rl.models.actor.WarmupDecayCooldownLR")
def test_actor_configure_optimizers(
    mock_scheduler, mock_get_model, dummy_config, dummy_mesh, dummy_checkpoint_manager, dummy_logger
):
    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
    mock_get_model.return_value = mock_model

    actor = Actor(dummy_config, dummy_mesh, dummy_checkpoint_manager, dummy_logger)
    actor.configure_optimizers(total_training_steps=100)

    assert isinstance(actor.optimizer, torch.optim.AdamW)
    mock_scheduler.assert_called_once()


@patch("phyagi.rl.models.actor._move_optimizer_state")
@patch("phyagi.rl.models.actor.get_model")
@pytest.mark.is_torch_gpu
def test_actor_on_gpu_manual_offload(
    mock_get_model, mock_move_opt_state, dummy_config, dummy_mesh, dummy_checkpoint_manager, dummy_logger
):
    dummy_config.manual_offload = True
    dummy_config.fsdp_offload = False

    mock_model = MagicMock()
    mock_get_model.return_value = mock_model

    actor = Actor(dummy_config, dummy_mesh, dummy_checkpoint_manager, dummy_logger)
    actor.optimizer = MagicMock()

    with actor.on_gpu():
        pass

    assert mock_move_opt_state.call_count == 2
