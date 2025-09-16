# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil

import pytest
import torch
from transformers import AutoTokenizer

from phyagi.datasets.train.lm.lm_dataset_provider import LMDatasetProvider
from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MixFormerSequentialConfig,
)
from phyagi.models.mixformer_sequential.modeling_mixformer_sequential import (
    MixFormerSequentialForCausalLM,
)
from phyagi.trainers.ds.ds_trainer import DsTrainer
from phyagi.trainers.ds.ds_training_args import DsTrainingArguments


@pytest.mark.is_mpi
@pytest.mark.is_torch_gpu
@pytest.mark.parametrize(
    ("steps", "expected_loss"),
    [
        pytest.param(50, 7.5, marks=pytest.mark.slow, id="50steps"),
        pytest.param(5000, 3.45, marks=pytest.mark.slowest, id="5000steps"),
    ],
)
def test_ds_train_regression(steps, expected_loss):
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
    dataset_provider = LMDatasetProvider.from_hub(
        dataset_path="wikitext",
        dataset_name="wikitext-103-raw-v1",
        tokenizer=tokenizer,
        cache_dir="test_tmp",
        seq_len=512,
    )

    train_dataset = dataset_provider.get_train_dataset()
    eval_dataset = dataset_provider.get_val_dataset()

    config = MixFormerSequentialConfig(n_embd=768, n_layer=8, n_positions=512)
    model = MixFormerSequentialForCausalLM(config=config)

    training_args = DsTrainingArguments(
        "test_tmp",
        max_steps=steps,
        logging_steps=steps,
        save_steps=steps,
        seed=42,
        eval_steps=steps,
        eval_max_steps=100,
        pipe_parallel_size=1,
    )
    training_args.ds_config["train_batch_size"] = 32

    trainer = DsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    assert (
        expected_loss - 0.5 <= round(trainer.client_state["log_history"][-1]["eval/0/loss"], 3) <= expected_loss + 0.5
    )

    shutil.rmtree("test_tmp", ignore_errors=True)
