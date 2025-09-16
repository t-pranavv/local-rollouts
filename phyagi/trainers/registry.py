# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset
from transformers import TrainingArguments

from phyagi.trainers.ds.ds_trainer import DsTrainer
from phyagi.trainers.ds.ds_training_args import DsTrainingArguments
from phyagi.trainers.hf.hf_trainer import HfTrainer
from phyagi.trainers.pl.pl_trainer import PlTrainer
from phyagi.trainers.pl.pl_training_args import PlTrainingArguments

TRAINING_ARGUMENTS = {
    "hf": TrainingArguments,
    "pl": PlTrainingArguments,
    "ds": DsTrainingArguments,
}

TRAINERS = {"hf": HfTrainer, "pl": PlTrainer, "ds": DsTrainer}


def get_training_args(output_dir: str, framework: str = "hf", **kwargs) -> Any:
    """Get training arguments for a given framework.

    Extra keyword arguments that are not shared across the frameworks
    are passed as keyword arguments to the framework-specific training arguments class.

    Args:
        output_dir: Output directory for checkpoints and predictions.
        framework: Framework to be used.

    Returns:
        Training arguments.

    """

    if framework not in TRAINING_ARGUMENTS:
        raise ValueError(f"`framework` must be one of {list(TRAINING_ARGUMENTS.keys())}, but got '{framework}'.")

    return TRAINING_ARGUMENTS[framework](output_dir, **kwargs)


def get_trainer(
    model: torch.nn.Module,
    framework: str = "hf",
    training_args: Optional[Any] = None,
    data_collator: Optional[Callable] = None,
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Optional[Dataset] = None,
    optimizers: Optional[Tuple[Optimizer, LRScheduler]] = (None, None),
    **kwargs,
) -> Any:
    """Get a trainer for a given framework.

    Extra keyword arguments that are not shared across the frameworks
    are passed as keyword arguments to the framework-specific trainer class.

    Args:
        model: Model to be trained or evaluated.
        framework: Framework to be used.
        training_args: Training arguments.
        data_collator: Collate function used for creating a batch from ``train_dataset`` and ``eval_dataset``.
        train_dataset: Dataset used for training.
        eval_dataset: Dataset used for evaluation.
        optimizers: Tuple of ``(optimizer, lr_scheduler)`` to be used for training.

    """

    if framework not in TRAINERS:
        raise ValueError(f"`framework` must be one of {list(TRAINERS.keys())}, but got '{framework}'.")

    return TRAINERS[framework](
        model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=optimizers,
        **kwargs,
    )
