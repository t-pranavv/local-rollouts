# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from phyagi.models.mixformer_sequential import MixFormerSequentialConfig
from phyagi.models.mixformer_sequential.blocks.mixers import get_mixer

config = MixFormerSequentialConfig(n_embd=1024, n_layer=2)
valid_mha_configs = [{}, {"n_head": 8}, {"n_head": 8, "n_head_kv": 1}, {"n_head": 8, "n_head_kv": 4}]
invalid_mha_configs = [{"n_head": 8, "head_dim": 64}, {"n_head": 8, "n_head_kv": 3}]


def _run_attn(mixer_cls: str, x: torch.tensor, fp16: bool = False, **extra_kwargs):
    for mha_config in valid_mha_configs:
        mixer = get_mixer(config, mixer_config={"mixer_cls": mixer_cls, **mha_config, **extra_kwargs}).to("cuda")

        if fp16:
            x = x.half()
            mixer = mixer.half()

        y = mixer(x)
        assert y.shape == x.shape

    for mha_config in invalid_mha_configs:
        try:
            mixer = get_mixer(config, mixer_config={"mixer_cls": mixer_cls, **mha_config, **extra_kwargs})
        except ValueError:
            pass


@pytest.mark.is_flash_attn
@pytest.mark.is_torch_gpu
def test_flash_mha():
    sample_input = torch.randn(2, config.n_positions, config.n_embd).to("cuda")
    _run_attn("mha", sample_input, flash_attn=True, fused_dense=True, flash_rotary=True, fp16=True)
