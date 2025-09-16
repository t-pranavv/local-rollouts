# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from phyagi.models.mixformer_sequential.blocks.mlps import get_mlp
from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MixFormerSequentialConfig,
)


def _run_mlp(config, mlp_cls, hidden_states):
    mlp = get_mlp(config, mlp_config={"mlp_cls": mlp_cls})
    output = mlp(hidden_states)
    assert output.shape == (2, 3, 1024)


def test_mlp():
    config = MixFormerSequentialConfig()
    hidden_states = torch.randn(2, 3, 1024)
    _run_mlp(config, "mlp", hidden_states)
    _run_mlp(config, "glu", hidden_states)


@pytest.mark.is_flash_attn
def test_flash_mlp():
    config = MixFormerSequentialConfig()
    hidden_states = torch.randn(2, 3, 1024)
    _run_mlp(config, "fused_mlp", hidden_states)
