# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil

import pytest
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from phyagi.rl.tuners.hf.hf_tuner import HfDPOTuner, HfGRPOTuner, HfSFTTuner
from phyagi.utils.hf_utils import AzureStorageRotateCheckpointMixin


@pytest.fixture
def dummy_model():
    return AutoModelForCausalLM.from_pretrained("gpt2")


@pytest.fixture
def dummy_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def dummy_dataset():
    return Dataset.from_dict(
        {
            # SFT requires `text` and `completion` columns (or `input_ids`)
            "text": ["Hello, "],
            "completion": ["world!"],
            # DPO requires `rejected`, `chosen` and `prompt`` column to be removed
            "prompt": ["Hello, "],
            "rejected": ["globe!"],
            "chosen": ["world!"],
        }
    )


def test_hf_sft_tuner(dummy_model, dummy_dataset):
    sft_config = SFTConfig(
        output_dir="test_tmp",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        bf16=False,
    )

    trainer = HfSFTTuner(model=dummy_model, args=sft_config, train_dataset=dummy_dataset)

    assert isinstance(trainer, SFTTrainer)
    assert isinstance(trainer, AzureStorageRotateCheckpointMixin)

    shutil.rmtree("test_tmp")


def test_hf_dpo_tuner(dummy_model, dummy_tokenizer, dummy_dataset):
    dpo_config = DPOConfig(
        output_dir="test_tmp",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        bf16=False,
    )

    trainer = HfDPOTuner(
        model=dummy_model,
        args=dpo_config,
        train_dataset=dummy_dataset,
        processing_class=dummy_tokenizer,
    )

    assert isinstance(trainer, DPOTrainer)
    assert isinstance(trainer, AzureStorageRotateCheckpointMixin)

    shutil.rmtree("test_tmp")


def test_hf_grpo_tuner(dummy_model, dummy_dataset):
    grpo_config = GRPOConfig(
        output_dir="test_tmp",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        bf16=False,
    )

    trainer = HfGRPOTuner(model=dummy_model, args=grpo_config, train_dataset=dummy_dataset, reward_funcs=[lambda x: 0])

    assert isinstance(trainer, GRPOTrainer)
    assert isinstance(trainer, AzureStorageRotateCheckpointMixin)

    shutil.rmtree("test_tmp")
