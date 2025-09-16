# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from phyagi.eval.generation import (
    HfGenerationEngine,
    LogLikelihoodPipelineEngine,
    LossPipelineEngine,
    TextGenerationPipelineEngine,
    example_generator,
)


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


@pytest.mark.parametrize(
    ("device"),
    [
        pytest.param("cpu", marks=pytest.mark.slowest, id="cpu"),
        pytest.param("cuda", marks=pytest.mark.is_torch_gpu, id="cuda"),
    ],
)
def test_hf_generation_engine(device):
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    engine = HfGenerationEngine()
    result = engine.generate(
        ["Once upon a time"],
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    assert len(result) == 1

    assert "responses" in result[0]
    assert isinstance(result[0]["responses"], list)
    assert len(result[0]["responses"]) > 0

    assert isinstance(result[0]["responses"][0], str)
    assert len(result[0]["responses"][0]) > 0


@pytest.mark.parametrize(
    ("device", "torch_dtype"),
    [
        pytest.param("cpu", torch.float32, marks=pytest.mark.slowest, id="cpu-float32"),
        pytest.param("cuda", torch.float32, marks=pytest.mark.is_torch_gpu, id="cuda-float32"),
        pytest.param("cuda", torch.float16, marks=pytest.mark.is_torch_gpu, id="cuda-float16"),
        pytest.param("cpu", torch.bfloat16, marks=pytest.mark.slowest, id="cpu-bfloat16"),
        pytest.param("cuda", torch.bfloat16, marks=pytest.mark.is_torch_gpu, id="cuda-bfloat16"),
    ],
)
def test_log_likelihood_pipeline_engine(device, torch_dtype):
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = model.to(device=device, dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    dataset = CustomDataset(
        [
            [
                {"text": "A long time ago in a galaxy far", "target": ", far away...", "label": "example_label"},
                {"text": "To be or not to be, that is the", "target": " question.", "label": "example_label"},
            ]
        ]
    )

    engine = LogLikelihoodPipelineEngine()
    result = engine.generate(
        dataset,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert "log_likelihoods" in result[0]
    assert all(isinstance(x, float) for x in result[0]["log_likelihoods"])


@pytest.mark.parametrize(
    ("device", "torch_dtype"),
    [
        pytest.param("cpu", torch.float32, marks=pytest.mark.slowest, id="cpu-float32"),
        pytest.param("cuda", torch.float32, marks=pytest.mark.is_torch_gpu, id="cuda-float32"),
        pytest.param("cuda", torch.float16, marks=pytest.mark.is_torch_gpu, id="cuda-float16"),
        pytest.param("cpu", torch.bfloat16, marks=pytest.mark.slowest, id="cpu-bfloat16"),
        pytest.param("cuda", torch.bfloat16, marks=pytest.mark.is_torch_gpu, id="cuda-bfloat16"),
    ],
)
def test_loss_pipeline_engine(device, torch_dtype):
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = model.to(device=device, dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    dataset = CustomDataset(
        [
            {"text": "A long time ago in a galaxy far, far away..."},
            {"text": "To be or not to be, that is the question."},
        ]
    )

    engine = LossPipelineEngine()
    result = engine.generate(
        dataset,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    assert len(result) == 2
    assert isinstance(result[0], dict)
    assert "loss" in result[0]


@pytest.mark.parametrize(
    ("device", "torch_dtype"),
    [
        pytest.param("cpu", torch.float32, marks=pytest.mark.slowest, id="cpu-float32"),
        pytest.param("cuda", torch.float32, marks=pytest.mark.is_torch_gpu, id="cuda-float32"),
        pytest.param("cuda", torch.float16, marks=pytest.mark.is_torch_gpu, id="cuda-float16"),
        pytest.param("cpu", torch.bfloat16, marks=pytest.mark.slowest, id="cpu-bfloat16"),
        pytest.param("cuda", torch.bfloat16, marks=pytest.mark.is_torch_gpu, id="cuda-bfloat16"),
    ],
)
def test_text_generation_pipeline_engine(device, torch_dtype):
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = model.to(device=device, dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    engine = TextGenerationPipelineEngine()
    result = engine.generate(
        [{"text": "Once upon a time,"}],
        model=model,
        tokenizer=tokenizer,
        device=device,
        example_generator_kwargs={"key": "value"},
        generation_config={"batch_size": 1},
    )

    assert len(result) == 1

    assert "responses" in result[0]
    assert isinstance(result[0]["responses"], list)
    assert len(result[0]["responses"]) > 0

    assert isinstance(result[0]["responses"][0], str)
    assert len(result[0]["responses"][0]) > 0


def test_example_generator():
    dataset = CustomDataset(["apple", "banana", "cherry"])

    def mapping_fn(x, prefix):
        return {"item": prefix + x.upper()}

    gen = example_generator(dataset, mapping_fn, prefix="Fruit: ")

    expected_results = [
        {"item": "Fruit: APPLE"},
        {"item": "Fruit: BANANA"},
        {"item": "Fruit: CHERRY"},
    ]

    for expected_result in expected_results:
        assert next(gen) == expected_result

    with pytest.raises(StopIteration):
        next(gen)
