# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from transformers import AutoTokenizer

from phyagi.eval.log_likelihood_pipeline import LogLikelihoodPipeline
from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MixFormerSequentialConfig,
)
from phyagi.models.mixformer_sequential.modeling_mixformer_sequential import (
    MixFormerSequentialForCausalLM,
)
from phyagi.utils.hf_utils import to_device_map


@pytest.mark.parametrize("device_map", ["cpu", pytest.param("cuda", marks=pytest.mark.is_torch_gpu)])
def test_log_likelihood_pipeline(device_map):
    config = MixFormerSequentialConfig(
        n_layer=2,
        architecture={"mixer": {"mixer_cls": "mha", "flash_attn": False, "flash_rotary": False, "fused_dense": False}},
    )
    model = to_device_map(MixFormerSequentialForCausalLM(config), device_map)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")

    pipeline = LogLikelihoodPipeline(model, tokenizer)

    source_text = "This is an"
    target_text = [" test sentence", " example of a test sentence"]
    label = "example_label"
    inputs = [[{"text": source_text, "target": target, "label": label} for target in target_text]]

    results = pipeline(inputs, return_inputs=True, use_amp=False)
    for result, _input in zip(results, inputs):
        assert isinstance(result, dict)
        assert "exact_matches" in result
        assert "log_likelihoods" in result
        assert isinstance(result["exact_matches"], list)
        assert isinstance(result["log_likelihoods"], list)
        assert len(result["exact_matches"]) == len(_input)
        assert len(result["log_likelihoods"]) == len(_input)
        assert isinstance(result["exact_matches"][0], bool)
        assert isinstance(result["log_likelihoods"][0], float)
