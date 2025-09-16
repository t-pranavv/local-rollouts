# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
from phyagi.trainers.pl.pl_trainer import PlTrainer
from phyagi.trainers.pl.pl_training_args import (
    PlLightningModuleArguments,
    PlTrainerArguments,
    PlTrainingArguments,
)


@pytest.mark.is_torch_gpu
def test_pl_trainer():
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

    trainer_args = PlTrainerArguments(precision="bf16-mixed", max_steps=1, log_every_n_steps=1, val_check_interval=1)
    lightning_module_args = PlLightningModuleArguments(optimizer={"type": "adamw"})

    training_args = PlTrainingArguments(
        "test_tmp",
        trainer=trainer_args,
        lightning_module=lightning_module_args,
        save_steps=1,
        seed=42,
        wandb=False,
    )
    trainer = PlTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.strategy.barrier()

    shutil.rmtree("test_tmp", ignore_errors=True)
