# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import pytest
import torch
from transformers import TrainingArguments

from phyagi.trainers.ds.ds_trainer import DsTrainer
from phyagi.trainers.ds.ds_training_args import DsTrainingArguments
from phyagi.trainers.hf.hf_trainer import HfTrainer
from phyagi.trainers.registry import get_trainer, get_training_args
from phyagi.utils.file_utils import get_full_path


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)


def test_get_training_args_hf():
    output_dir = "output"
    hf_kwargs = {"num_train_epochs": 3}

    training_args = get_training_args(output_dir, "hf", **hf_kwargs)
    assert isinstance(training_args, TrainingArguments)
    assert training_args.output_dir == output_dir
    assert training_args.num_train_epochs == hf_kwargs["num_train_epochs"]


@pytest.mark.is_mpi
@pytest.mark.is_torch_gpu
def test_get_training_args_ds():
    output_dir = "output"
    ds_kwargs = {"num_train_epochs": 3}

    training_args = get_training_args(output_dir, "ds", **ds_kwargs)
    assert isinstance(training_args, DsTrainingArguments)
    assert training_args.output_dir == get_full_path(output_dir)
    assert training_args.num_train_epochs == ds_kwargs["num_train_epochs"]


def test_get_trainer_hf():
    args = TrainingArguments("output")
    trainer = get_trainer(Model(), "hf", args)
    assert isinstance(trainer, HfTrainer)


@pytest.mark.is_mpi
@pytest.mark.is_torch_gpu
def test_get_trainer_ds():
    os.environ["LOCAL_RANK"] = "0"

    args = DsTrainingArguments("output", wandb=False)
    trainer = get_trainer(Model(), "ds", args)
    assert isinstance(trainer, DsTrainer)
