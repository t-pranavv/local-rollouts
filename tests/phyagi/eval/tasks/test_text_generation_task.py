# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch
from transformers import AutoTokenizer

from phyagi.eval.tasks.human_eval import HumanEval
from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MixFormerSequentialConfig,
)
from phyagi.models.mixformer_sequential.modeling_mixformer_sequential import (
    MixFormerSequentialForCausalLM,
)
from phyagi.utils.hf_utils import to_device_map


@pytest.mark.parametrize(
    ("device_map", "torch_dtype"),
    [
        pytest.param("cpu", torch.float32, marks=pytest.mark.slowest, id="cpu-float32"),
        pytest.param("cuda", torch.float32, marks=pytest.mark.is_torch_gpu, id="cuda-float32"),
        pytest.param("cuda", torch.float16, marks=pytest.mark.is_torch_gpu, id="cuda-float16"),
        pytest.param("cpu", torch.bfloat16, marks=pytest.mark.slowest, id="cpu-bfloat16"),
        pytest.param("cuda", torch.bfloat16, marks=pytest.mark.is_torch_gpu, id="cuda-bfloat16"),
    ],
)
def test_text_generation_task(device_map, torch_dtype):
    config = MixFormerSequentialConfig(
        n_embd=128,
        n_layer=2,
        architecture={"mixer": {"mixer_cls": "mha", "flash_attn": False, "flash_rotary": False, "fused_dense": False}},
    )
    model = to_device_map(MixFormerSequentialForCausalLM(config).to(torch_dtype), device_map)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")

    human_eval_results = HumanEval.run(model, tokenizer, n_examples=1)
    assert isinstance(human_eval_results, dict)
    assert "pass@1" in human_eval_results.keys()
