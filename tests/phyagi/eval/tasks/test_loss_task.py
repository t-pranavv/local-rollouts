# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from phyagi.eval.tasks.loss import LossHFHubDataset, LossNumpyDataset
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
        pytest.param("cpu", torch.float32, id="cpu-float32"),
        pytest.param("cuda", torch.float32, marks=pytest.mark.is_torch_gpu, id="cuda-float32"),
        pytest.param("cuda", torch.float16, marks=pytest.mark.is_torch_gpu, id="cuda-float16"),
        pytest.param("cpu", torch.bfloat16, id="cpu-bfloat16"),
        pytest.param("cuda", torch.bfloat16, marks=pytest.mark.is_torch_gpu, id="cuda-bfloat16"),
    ],
)
def test_loss_hf_hub_task(device_map, torch_dtype):
    config = MixFormerSequentialConfig(
        n_embd=128,
        n_layer=2,
        architecture={"mixer": {"mixer_cls": "mha", "flash_attn": False, "flash_rotary": False, "fused_dense": False}},
    )
    model = to_device_map(MixFormerSequentialForCausalLM(config).to(torch_dtype), device_map)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")

    loss_hf_hub_results = LossHFHubDataset.run(
        model, tokenizer, dataset_path="wikitext", dataset_name="wikitext-2-raw-v1", split="test", n_examples=1
    )
    assert isinstance(loss_hf_hub_results, dict)
    assert "loss" in loss_hf_hub_results.keys()
    assert "ppl" in loss_hf_hub_results.keys()


@pytest.mark.parametrize(
    ("device_map", "torch_dtype"),
    [
        pytest.param("cpu", torch.float32, id="cpu-float32"),
        pytest.param("cuda", torch.float32, marks=pytest.mark.is_torch_gpu, id="cuda-float32"),
        pytest.param("cuda", torch.float16, marks=pytest.mark.is_torch_gpu, id="cuda-float16"),
        pytest.param("cpu", torch.bfloat16, id="cpu-bfloat16"),
        pytest.param("cuda", torch.bfloat16, marks=pytest.mark.is_torch_gpu, id="cuda-bfloat16"),
    ],
)
def test_loss_numpy_task(device_map, torch_dtype):
    npy_file_path = "tmp.npy"

    input_ids = np.zeros(64, dtype=np.int32)
    np.save(npy_file_path, input_ids)

    config = MixFormerSequentialConfig(
        n_embd=128,
        n_layer=2,
        architecture={"mixer": {"mixer_cls": "mha", "flash_attn": False, "flash_rotary": False, "fused_dense": False}},
    )
    model = to_device_map(MixFormerSequentialForCausalLM(config).to(torch_dtype), device_map)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")

    loss_numpy_results = LossNumpyDataset.run(model, tokenizer, npy_file_path=npy_file_path, seq_len=16)
    assert isinstance(loss_numpy_results, dict)
    assert "loss" in loss_numpy_results.keys()
    assert "ppl" in loss_numpy_results.keys()

    os.remove("tmp.npy")
