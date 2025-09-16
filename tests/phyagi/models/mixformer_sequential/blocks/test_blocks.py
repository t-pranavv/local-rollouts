# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from phyagi.models.mixformer_sequential.blocks.parallel import ParallelBlock
from phyagi.models.mixformer_sequential.blocks.sequential import SequentialBlock
from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MixFormerSequentialConfig,
)


@pytest.mark.is_flash_attn
@pytest.mark.is_torch_gpu
def test_parallel_block():
    config = MixFormerSequentialConfig()
    hidden_states = torch.randn(1, 1, config.n_embd).to("cuda")
    block = ParallelBlock(config).to("cuda")

    with torch.autocast("cuda", enabled=True):
        output = block(hidden_states)
    assert output[0].shape == (1, 1, config.n_embd)


@pytest.mark.is_flash_attn
@pytest.mark.is_torch_gpu
def test_sequential_block():
    config = MixFormerSequentialConfig()
    hidden_states = torch.randn(1, 1, config.n_embd).to("cuda")
    block = SequentialBlock(config).to("cuda")

    with torch.autocast("cuda", enabled=True):
        output = block(hidden_states)
    assert output[0].shape == (1, 1, config.n_embd)
