# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from transformers import AutoTokenizer

from phyagi.eval.text_generation_pipeline import TextGenerationPipeline
from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MixFormerSequentialConfig,
)
from phyagi.models.mixformer_sequential.modeling_mixformer_sequential import (
    MixFormerSequentialForCausalLM,
)
from phyagi.utils.hf_utils import to_device_map


@pytest.mark.parametrize("device_map", ["cpu", pytest.param("cuda", marks=pytest.mark.is_torch_gpu)])
def test_text_generation_pipeline(device_map):
    config = MixFormerSequentialConfig(
        n_layer=2,
        architecture={"mixer": {"mixer_cls": "mha", "flash_attn": False, "flash_rotary": False, "fused_dense": False}},
    )
    model = to_device_map(MixFormerSequentialForCausalLM(config), device_map)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
    pipeline = TextGenerationPipeline(model, tokenizer)

    source_text = "This is an example of a test sentence."
    label = "example_label"
    inputs = {"text": source_text, "label": label}

    results = pipeline(inputs, n_samples=1, return_inputs=True, use_attention_mask=True, max_length=20)
    assert isinstance(results, dict)
    assert "responses" in results
    assert isinstance(results["responses"], list)
    assert len(results["responses"]) == 1
    assert isinstance(results["responses"][0], str)
    assert len(results["responses"][0]) > 0
