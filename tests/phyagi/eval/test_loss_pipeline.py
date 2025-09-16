# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from transformers import AutoTokenizer

from phyagi.eval.loss_pipeline import LossPipeline
from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MixFormerSequentialConfig,
)
from phyagi.models.mixformer_sequential.modeling_mixformer_sequential import (
    MixFormerSequentialForCausalLM,
)
from phyagi.utils.hf_utils import to_device_map


@pytest.mark.parametrize("device_map", [pytest.param("cpu"), pytest.param("cuda", marks=pytest.mark.is_torch_gpu)])
def test_loss_pipeline(device_map):
    config = MixFormerSequentialConfig(
        n_layer=2,
        architecture={"mixer": {"mixer_cls": "mha", "flash_attn": False, "flash_rotary": False, "fused_dense": False}},
    )
    model = to_device_map(MixFormerSequentialForCausalLM(config), device_map)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
    pipeline = LossPipeline(model, tokenizer)

    source_text = "This is an example of a test sentence."
    inputs = {"text": source_text}

    results = pipeline(inputs, use_amp=False, shift_labels=True)
    assert isinstance(results, dict)
    assert "loss" in results
    assert isinstance(results["loss"], float)
