# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch.utils.data import Dataset
from trl import DPOConfig, GRPOConfig, SFTConfig

from phyagi.rl.tuners.grpo.grpo_config import RayGRPOConfig
from phyagi.rl.tuners.hf.hf_tuner import HfDPOTuner, HfGRPOTuner, HfSFTTuner
from phyagi.rl.tuners.isft.isft_config import RayISFTConfig
from phyagi.utils.import_utils import is_vllm_available

RayGRPOTuner = None
RayISFTTuner = None
if is_vllm_available():
    from phyagi.rl.tuners.grpo.grpo_tuner import RayGRPOTuner
    from phyagi.rl.tuners.isft.isft_tuner import RayISFTTuner

TUNING_ARGUMENTS = {
    "hf": {
        "sft": SFTConfig,
        "dpo": DPOConfig,
        "grpo": GRPOConfig,
    },
    "ray": {
        "isft": RayISFTConfig,
        "grpo": RayGRPOConfig,
    },
}

TUNERS = {
    "hf": {
        "sft": HfSFTTuner,
        "dpo": HfDPOTuner,
        "grpo": HfGRPOTuner,
    },
    "ray": {
        "isft": RayISFTTuner,
        "grpo": RayGRPOTuner,
    },
}


def get_tuning_args(output_dir: str, framework: str = "hf", task: str = "sft", **kwargs) -> Any:
    """Get tuning arguments for a given framework and task.

    Extra keyword arguments that are not shared across the frameworks
    are passed as keyword arguments to the framework-specific tuning arguments class.

    Args:
        output_dir: Output directory for checkpoints and predictions.
        framework: Framework to be used.
        task: Task to be used.

    Returns:
        Tuner arguments.

    """

    if framework not in TUNING_ARGUMENTS:
        raise ValueError(f"`framework` must be one of {list(TUNING_ARGUMENTS.keys())}, but got '{framework}'.")

    framework_args = TUNING_ARGUMENTS[framework]
    if task not in framework_args:
        raise ValueError(f"`task` must be one of {list(framework_args.keys())}, but got '{task}'.")

    return framework_args[task](output_dir, **kwargs)


def get_tuner(
    framework: str = "hf",
    task: str = "sft",
    model: Optional[torch.nn.Module] = None,
    tuning_args: Optional[Any] = None,
    data_collator: Optional[Callable] = None,
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Optional[Dataset] = None,
    **kwargs,
) -> Any:
    """Get a tuner for a given framework and task.

    Extra keyword arguments that are not shared across the frameworks
    are passed as keyword arguments to the framework-specific tuner class.

    Args:
        framework: Framework to be used.
        task: Task to be used.
        model: Model to be tuned or evaluated.
        tuning_args: Tuner arguments.
        data_collator: Collate function used for creating a batch from ``train_dataset`` and ``eval_dataset``.
        train_dataset: Dataset used for tuning.
        eval_dataset: Dataset used for evaluation.

    """

    if framework not in TUNERS:
        raise ValueError(f"`framework` must be one of {list(TUNERS.keys())}, but got '{framework}'.")

    framework_tuners = TUNERS[framework]
    if task not in framework_tuners:
        raise ValueError(f"`task` must be one of {list(framework_tuners.keys())}, but got '{task}'.")

    tuner_cls = framework_tuners[task]
    if tuner_cls is None:
        raise ValueError(f"'{framework}' -> '{task}' is not available due to a missing dependency.")

    return tuner_cls(
        model=model,
        args=tuning_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **kwargs,
    )
