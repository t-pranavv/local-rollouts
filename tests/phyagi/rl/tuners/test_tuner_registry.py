# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from datasets import Dataset
from transformers import AutoModelForCausalLM
from trl import SFTConfig

from phyagi.rl.tuners.grpo.grpo_config import RayGRPOConfig
from phyagi.rl.tuners.hf.hf_tuner import HfSFTTuner
from phyagi.rl.tuners.registry import get_tuner, get_tuning_args
from phyagi.utils.file_utils import get_full_path


@pytest.fixture
def dummy_dataset():
    return Dataset.from_dict(
        {
            "text": ["Hello, "],
            "completion": ["world!"],
        }
    )


def test_get_tuning_args_hf_sft():
    output_dir = "output"
    hf_kwargs = {"num_train_epochs": 3, "bf16": False}

    tuning_args = get_tuning_args(output_dir, framework="hf", task="sft", **hf_kwargs)
    assert isinstance(tuning_args, SFTConfig)
    assert tuning_args.output_dir == output_dir
    assert tuning_args.num_train_epochs == hf_kwargs["num_train_epochs"]


def test_get_tuner_hf_sft(dummy_dataset):
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    args = SFTConfig("output", bf16=False)

    trainer = get_tuner(framework="hf", task="sft", model=model, tuning_args=args, train_dataset=dummy_dataset)
    assert isinstance(trainer, HfSFTTuner)


def test_get_tuning_args_ray_grpo():
    output_dir = "output"
    ray_kwargs = {
        "n_nodes": 1,
        "n_gpus_per_node": 1,
        "max_steps": 1,
    }

    tuning_args = get_tuning_args(output_dir, framework="ray", task="grpo", **ray_kwargs)
    assert isinstance(tuning_args, RayGRPOConfig)
    assert tuning_args.output_dir == get_full_path(output_dir)
    assert tuning_args.n_nodes == ray_kwargs["n_nodes"]
    assert tuning_args.n_gpus_per_node == ray_kwargs["n_gpus_per_node"]
    assert tuning_args.max_steps == ray_kwargs["max_steps"]


@pytest.mark.is_vllm
def test_get_tuner_ray_grpo():
    from phyagi.rl.tuners.grpo.grpo_tuner import RayGRPOTuner

    args = RayGRPOConfig("output", n_nodes=1, n_gpus_per_node=1, max_steps=1)

    trainer = get_tuner(framework="ray", task="grpo", tuning_args=args)
    assert isinstance(trainer, RayGRPOTuner)
