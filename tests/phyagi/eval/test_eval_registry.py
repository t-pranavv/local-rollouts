# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from transformers import AutoTokenizer

from phyagi.eval.registry import run_task
from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MixFormerSequentialConfig,
)
from phyagi.models.mixformer_sequential.modeling_mixformer_sequential import (
    MixFormerSequentialForCausalLM,
)
from phyagi.utils.hf_utils import to_device_map


@pytest.mark.is_torch_gpu
@pytest.mark.slow
def test_run_task():
    config = MixFormerSequentialConfig(
        n_layer=2,
        architecture={"mixer": {"mixer_cls": "mha", "flash_attn": False, "flash_rotary": False, "fused_dense": False}},
    )
    model = to_device_map(MixFormerSequentialForCausalLM(config), "cuda")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
    task_name = "arc_easy"

    result = run_task(task_name, model, tokenizer)
    assert isinstance(result, dict)
    assert "accuracy" in result
    assert isinstance(result["accuracy"], float)
