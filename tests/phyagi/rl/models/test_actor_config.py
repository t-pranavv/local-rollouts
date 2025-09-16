# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from phyagi.rl.models.actor_config import ActorConfig


def test_actor_config():
    config = ActorConfig()
    assert config.model == {}
    assert config.use_meta_tensor is False
    assert config.optimizer["lr"] == 1e-6
    assert config.optimizer["betas"] == (0.9, 0.999)
    assert config.optimizer["weight_decay"] == 1e-2
    assert config.scheduler["warmup_num_steps"] == 500
    assert config.gradient_clipping == 1.0
    assert config.manual_offload is False
    assert config.fsdp_offload is False
    assert config.activation_checkpointing is False
    assert config.dtype == "bfloat16"
    assert config.adam_8bit is False

    custom_config = ActorConfig(
        model={"type": "transformer", "layers": 12},
        optimizer={"lr": 1e-4, "betas": (0.95, 0.98), "weight_decay": 0.01},
        scheduler={"warmup_num_steps": 1000, "decay": True},
        gradient_clipping=0.5,
        manual_offload=False,
        fsdp_offload=True,
        activation_checkpointing=True,
        dtype="float32",
        adam_8bit=True,
    )
    assert custom_config.model == {"type": "transformer", "layers": 12}
    assert custom_config.optimizer["lr"] == 1e-4
    assert custom_config.optimizer["betas"] == (0.95, 0.98)
    assert custom_config.optimizer["weight_decay"] == 0.01
    assert custom_config.scheduler["warmup_num_steps"] == 1000
    assert custom_config.scheduler["decay"] is True
    assert custom_config.gradient_clipping == 0.5
    assert custom_config.manual_offload is False
    assert custom_config.fsdp_offload is True
    assert custom_config.activation_checkpointing is True
    assert custom_config.dtype == "float32"
    assert custom_config.adam_8bit is True

    with pytest.raises(NotImplementedError):
        custom_config = ActorConfig(use_meta_tensor=True)
    with pytest.raises(ValueError):
        custom_config = ActorConfig(manual_offload=True, fsdp_offload=True)
