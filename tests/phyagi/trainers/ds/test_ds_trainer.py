# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil

import pytest
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
    "pipe_parallel_size",
    [
        pytest.param(0, id="pp-0"),
        pytest.param(1, id="pp-1"),
    ],
)
def test_ds_trainer(pipe_parallel_size):
    os.environ["LOCAL_RANK"] = "0"

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1")
    dataset_provider = LMDatasetProvider.from_hub(
        dataset_path="glue",
        dataset_name="sst2",
        mapping_column_name="sentence",
        tokenizer=tokenizer,
        cache_dir="test_tmp",
        seq_len=2048,
    )

    train_dataset = dataset_provider.get_train_dataset()
    eval_dataset = dataset_provider.get_val_dataset()

    config = MixFormerSequentialConfig(
        vocab_size=50304,
        n_positions=2048,
        n_embd=128,
        n_layer=4,
        n_head=8,
        rotary_dim=16,
        use_fused_mlp=False,
    )
    model = MixFormerSequentialForCausalLM(config=config)

    training_args = DsTrainingArguments(
        "test_tmp",
        max_steps=1,
        logging_steps=1,
        save_steps=1,
        seed=42,
        eval_steps=1,
        eval_max_steps=1,
        pipe_parallel_size=pipe_parallel_size,
        wandb=False,
    )
    trainer = DsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    assert len(trainer.client_state["log_history"]) > 0

    shutil.rmtree("test_tmp", ignore_errors=True)
