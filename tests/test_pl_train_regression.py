# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import shutil

import pandas as pd
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
from phyagi.trainers.pl.pl_trainer import PlTrainer
from phyagi.trainers.pl.pl_training_args import (
    PlLightningModuleArguments,
    PlTrainerArguments,
    PlTrainingArguments,
)


@pytest.mark.is_torch_gpu
@pytest.mark.parametrize(
    ("steps", "expected_loss"),
    [
        pytest.param(50, 7, marks=pytest.mark.slow, id="50steps"),
        pytest.param(5000, 3.5, marks=pytest.mark.slowest, id="5000steps"),
    ],
)
def test_pl_train_regression(steps, expected_loss):
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

    training_args = PlTrainingArguments(
        "test_tmp",
        save_steps=steps,
        seed=42,
        trainer=PlTrainerArguments(
            precision=16,
            max_steps=steps,
            accumulate_grad_batches=8,
            log_every_n_steps=steps,
            limit_val_batches=100,
            val_check_interval=steps,
        ),
        lightning_module=PlLightningModuleArguments(
            optimizer={"type": "adamw", "lr": 1.8e-3},
        ),
    )

    trainer = PlTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    df = pd.read_csv("test_tmp/metrics.csv")
    df = df[df["eval/loss"].notnull()]

    assert expected_loss - 0.5 <= round(df["eval/loss"].iloc[-1], 3) <= expected_loss + 0.5

    shutil.rmtree("test_tmp", ignore_errors=True)
