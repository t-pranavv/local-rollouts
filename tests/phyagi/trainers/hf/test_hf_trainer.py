# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, TrainingArguments

from phyagi.datasets.train.lm.lm_dataset_provider import LMDatasetProvider
from phyagi.models.mixformer_sequential.configuration_mixformer_sequential import (
    MixFormerSequentialConfig,
)
from phyagi.models.mixformer_sequential.modeling_mixformer_sequential import (
    MixFormerSequentialForCausalLM,
)
from phyagi.trainers.hf.hf_trainer import HfTrainer

os.environ["WANDB_MODE"] = "disabled"


def test_hf_trainer_rotate_checkpoints():
    model = torch.nn.Linear(10, 5)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        checkpoint_1 = temp_path / "checkpoint-1"
        checkpoint_1.mkdir()
        checkpoint_2 = temp_path / "checkpoint-2"
        checkpoint_2.mkdir()
        checkpoint_3 = temp_path / "checkpoint-3"
        checkpoint_3.mkdir()

        # Nothing happens when `save_total_limit` is None or 0
        args = TrainingArguments("test_tmp", save_total_limit=None, load_best_model_at_end=False)
        trainer = HfTrainer(model, args=args)
        trainer._rotate_checkpoints(output_dir=temp_path)
        assert checkpoint_1.exists()
        assert checkpoint_2.exists()
        assert checkpoint_3.exists()

        args = TrainingArguments("test_tmp", save_total_limit=0, load_best_model_at_end=False)
        trainer = HfTrainer(model, args=args)
        trainer._rotate_checkpoints(output_dir=temp_path)
        assert checkpoint_1.exists()
        assert checkpoint_2.exists()
        assert checkpoint_3.exists()

        # Only the oldest checkpoint is deleted
        args = TrainingArguments("test_tmp", save_total_limit=2, load_best_model_at_end=False)
        trainer = HfTrainer(model, args=args)
        trainer._rotate_checkpoints(output_dir=temp_path)
        assert not checkpoint_1.exists()
        assert checkpoint_2.exists()
        assert checkpoint_3.exists()

    shutil.rmtree("test_tmp")


@pytest.mark.is_torch_gpu
def test_hf_trainer():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
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
        architecture={
            "block_cls": "parallel",
            "mixer": {"mixer_cls": "mha", "flash_attn": False},
        },
    )
    model = MixFormerSequentialForCausalLM(config=config)

    training_args = TrainingArguments(
        "test_tmp",
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        eval_strategy="steps",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1.8e-3,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        max_steps=1,
        warmup_steps=0,
        logging_steps=1,
        save_steps=1,
        eval_steps=1,
        seed=42,
    )
    trainer = HfTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer_output = trainer.train()
    assert trainer_output.global_step == 1
    assert trainer_output.training_loss >= 0.0

    shutil.rmtree("test_tmp", ignore_errors=True)
