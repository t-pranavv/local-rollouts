# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import deepspeed
import torch
from omegaconf import OmegaConf

from phyagi.utils.config import load_config
from phyagi.utils.file_utils import get_full_path


@dataclass
class DsTrainingArguments:
    """Define arguments used in the DeepSpeed training pipeline.

    Args:
        output_dir: Output folder where checkpoints and states will be written.
        ds_config: DeepSpeed configuration (dictionary or path to JSON file).
        do_eval: Whether to run evaluation on the validation set.
        do_final_eval: Whether to run last step evaluation on the validation set.
        train_batch_size_init_rampup: Training batch size initial rampup value.
            If set to ``0``, will not use rampup.
        train_batch_size_per_rampup: Training batch size increase per rampup.
            If ``train_batch_size_init_rampup > 0``, it must be set to ``> 0``.
        rampup_steps: Number of steps between rampups.
            If ``train_batch_size_init_rampup > 0``, it must be set to ``>= 1``.
        num_train_epochs: Number of training epochs.
            If type is ``float``, will use the decimal part of the last epoch.
        max_steps: Maximum number of training steps.
            If set to ``> 0``, will override ``num_train_epochs``.
        logging_steps: Number of steps between logs.
            If set to range ``[0, 1)``, will be interpreted as a ratio of total training steps.
        save_steps: Number of steps between checkpoints.
            If set to range ``[0, 1)``, will be interpreted as a ratio of total training steps.
        save_final_checkpoint: Whether to save last step checkpoint.
        eval_steps: Number of steps between evaluations.
            If set to range ``[0, 1)``, will be interpreted as a ratio of total training steps.
        eval_max_steps: Maximum number of evaluation steps.
        seed: Random seed.
        pipe_parallel_size: Size of pipeline parallelism.
        pipe_parallel_partition_method: Partition method for pipeline parallelism.
        pipe_parallel_activation_checkpoint_steps: Number of steps between pipeline parallelism activation checkpoins.
        tensor_parallel_size: Size of tensor parallelism.
        context_parallel_size: Size of context parallelism.
        batch_tracker: Whether to use batch tracker.
        batch_tracker_save_steps: Number of steps between saving batch trackings.
        dataloader_shuffle: Whether to shuffle the data loader (training).
        eval_dataloader_shuffle: Whether to shuffle the data loader (evaluation).
        dataloader_pin_memory: Whether to pin the data loader memory.
        dataloader_num_workers: Number of subprocesses to use for data loading.
        dataloader_prefetch_factor: Queue size for prefetch (per worker).
        load_checkpoint_num_tries: Number of tries for loading a checkpoint.
        backend: Distributed training backend.
        timeout: Timeout in seconds for operations executed against the process group.
        log_dir: Directory to save logs. If not provided, will use ``output_dir``.
        mlflow: Whether to enable MLflow logging.
        wandb: Whether to enable Weights & Biases logging.
        wandb_api_key: Weights & Biases API key.
        wandb_host: Weights & Biases host name.
        tensorboard: Whether to enable TensorBoard logging.

    """

    output_dir: Union[str, Path] = field(
        metadata={"help": "Output folder where checkpoints and states will be written."}
    )

    ds_config: Union[Dict[str, Any], str] = field(
        default_factory=dict, metadata={"help": "DeepSpeed configuration (dictionary or path to JSON file)."}
    )

    do_eval: bool = field(default=True, metadata={"help": "Whether to run evaluation on the validation set."})

    do_final_eval: bool = field(
        default=False, metadata={"help": "Whether to run last step evaluation on the validation set."}
    )

    train_batch_size_init_rampup: int = field(default=0, metadata={"help": "Training batch size initial rampup value."})

    train_batch_size_per_rampup: int = field(default=0, metadata={"help": "Training batch size increase per rampup."})

    rampup_steps: int = field(default=0, metadata={"help": "Number of steps between rampups."})

    num_train_epochs: Union[int, float] = field(
        default=1,
        metadata={"help": "Number of training epochs."},
    )

    max_steps: int = field(
        default=-1,
        metadata={"help": "Maximum number of training steps."},
    )

    logging_steps: Union[int, float] = field(
        default=10,
        metadata={"help": "Number of steps between logs."},
    )

    save_steps: Union[int, float] = field(
        default=500,
        metadata={"help": "Number of steps between checkpoints."},
    )

    save_final_checkpoint: bool = field(default=False, metadata={"help": "Whether to save last step checkpoint."})

    eval_steps: Union[int, float] = field(
        default=500,
        metadata={"help": "Number of steps between evaluations."},
    )

    eval_max_steps: int = field(default=None, metadata={"help": "Number of maximum steps during evaluation."})

    seed: int = field(default=42, metadata={"help": "Random seed."})

    pipe_parallel_size: int = field(default=1, metadata={"help": "Size of pipeline parallelism."})

    pipe_parallel_partition_method: Union[List[int], str] = field(
        default="parameters", metadata={"help": "Partition method for pipeline parallelism."}
    )

    pipe_parallel_activation_checkpoint_steps: int = field(
        default=0, metadata={"help": "Number of steps between pipeline parallelism activation checkpoins."}
    )

    tensor_parallel_size: int = field(default=1, metadata={"help": "Size of tensor parallelism."})

    context_parallel_size: int = field(default=1, metadata={"help": "Size of context parallelism."})

    batch_tracker: bool = field(default=False, metadata={"help": "Whether to use batch tracker."})

    batch_tracker_save_steps: Optional[Union[int, float]] = field(
        default=None, metadata={"help": "Number of steps between saving batch trackings."}
    )

    dataloader_shuffle: bool = field(default=True, metadata={"help": "Whether to shuffle the data loader (training)."})

    eval_dataloader_shuffle: bool = field(
        default=True, metadata={"help": "Whether to shuffle the data loader (evaluation)."}
    )

    dataloader_pin_memory: bool = field(default=True, metadata={"help": "Whether to pin the data loader memory."})

    dataloader_num_workers: int = field(default=0, metadata={"help": "Number of subprocesses to use for data loading."})

    dataloader_prefetch_factor: int = field(default=None, metadata={"help": "Queue size for prefetch (per worker)."})

    load_checkpoint_num_tries: int = field(default=1, metadata={"help": "Number of tries for loading a checkpoint."})

    backend: Optional[str] = field(default=None, metadata={"help": "Distributed training backend."})

    timeout: int = field(
        default=1800, metadata={"help": "Timeout in seconds for operations executed against the process group."}
    )

    log_dir: Union[str, Path] = field(default=None, metadata={"help": "Directory to save logs."})

    mlflow: bool = field(default=False, metadata={"help": "Whether to enable MLflow logging."})

    wandb: bool = field(default=False, metadata={"help": "Whether to enable Weights & Biases logging."})

    wandb_api_key: str = field(default=None, metadata={"help": "Weights & Biases API key."})

    wandb_host: str = field(default=None, metadata={"help": "Weights & Biases host name."})

    tensorboard: bool = field(default=False, metadata={"help": "Whether to enable TensorBoard logging."})

    def _create_default_deepspeed_config(self) -> Dict[str, Any]:
        return {
            "train_batch_size": 1024,
            "train_micro_batch_size_per_gpu": 4,
            "fp16": {"enabled": True, "initial_scale_power": 12},
            "zero_optimization": {"stage": 0},
            "optimizer": {"type": "AdamW", "params": {"lr": 1.8e-3, "betas": [0.9, 0.95], "eps": 1e-7}},
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {"warmup_min_lr": 0.0, "warmup_max_lr": 1.8e-3, "warmup_type": "linear"},
            },
            "steps_per_print": 1000,
        }

    def __post_init__(self) -> None:
        self.output_dir = get_full_path(self.output_dir, create_folder=True)
        self.log_dir = get_full_path(self.log_dir, create_folder=True) if self.log_dir is not None else self.output_dir

        # Use OmegaConf as the underlying `ds_config` structure for easier manipulation
        self.ds_config = self.ds_config or self._create_default_deepspeed_config()
        self.ds_config = load_config(self.ds_config, use_native_types=False)
        if self.train_batch_size is None:
            raise ValueError("`train_batch_size` must be defined in `ds_config`, but got None.")
        if self.train_micro_batch_size_per_gpu is None:
            raise ValueError("`train_micro_batch_size_per_gpu` must be defined in `ds_config`, but got None.")

        # Ensure that batch size rampup-based arguments are valid
        if self.train_batch_size_init_rampup > self.train_batch_size:
            raise ValueError(
                f"`train_batch_size_init_rampup` must be <= {self.train_batch_size}, but got {self.train_batch_size_init_rampup}."
            )
        if self.train_batch_size_init_rampup > 0:
            if self.train_batch_size_per_rampup <= 0:
                raise ValueError(
                    f"`train_batch_size_per_rampup` must be > 0, but got {self.train_batch_size_per_rampup}."
                )
            if self.rampup_steps < 1:
                raise ValueError(f"`rampup_steps` must be >= 1, but got {self.rampup_steps}.")

        # When using AMD-based GPUs, fused optimizers are not supported
        if torch.version.hip is not None:
            if OmegaConf.select(self.ds_config, "optimizer.type") == "AdamW":
                OmegaConf.update(self.ds_config, "optimizer.params.torch_adam", True)

        # Ensure `logging_steps`, `save_steps`, and `eval_steps` are integers
        # if they were not provided as ratios, e.g., [0, 1)
        if self.logging_steps > 1:
            self.logging_steps = int(self.logging_steps)
        if self.save_steps > 1:
            self.save_steps = int(self.save_steps)
        if self.eval_steps > 1:
            self.eval_steps = int(self.eval_steps)
        if self.batch_tracker_save_steps is None:
            self.batch_tracker_save_steps = self.logging_steps

        torch.manual_seed(self.seed)
        deepspeed.runtime.utils.set_random_seed(self.seed)

        deepspeed.init_distributed(dist_backend=self.backend, timeout=datetime.timedelta(seconds=self.timeout))
        self.world_size = deepspeed.comm.get_world_size()
        self.rank = deepspeed.comm.get_rank()
        self.local_rank = deepspeed.comm.get_local_rank()

        if self.pipe_parallel_size <= 0:
            self.pipe_parallel_size = 1
        if self.tensor_parallel_size <= 0:
            self.tensor_parallel_size = 1
        if self.context_parallel_size <= 0:
            self.context_parallel_size = 1

        if self.world_size % (self.pipe_parallel_size * self.tensor_parallel_size * self.context_parallel_size) != 0:
            raise ValueError(
                f"Total number of GPUs must be divisible by `pipe_parallel_size * tensor_parallel_size * context_parallel_size`, "
                f"but got {self.world_size} and {self.pipe_parallel_size * self.tensor_parallel_size * self.context_parallel_size}."
            )
        self.data_parallel_size = (
            self.world_size // self.pipe_parallel_size // self.tensor_parallel_size // self.context_parallel_size
        )

        if self.pipe_parallel_size > 1 and self.context_parallel_size > 1:
            raise ValueError(
                f"`pipe_parallel_size` and `context_parallel_size` cannot be used together, but got {self.pipe_parallel_size} and {self.context_parallel_size}."
            )
        if self.tensor_parallel_size > 1:
            raise ValueError(
                f"`tensor_parallel_size` has not been implemented yet, but got {self.tensor_parallel_size}."
            )
        if self.context_parallel_size > 1 and OmegaConf.select(self.ds_config, "zero_optimization.stage") == 3:
            raise ValueError(
                f"`context_parallel_size` cannot be used with ZeRO-3, but got {self.context_parallel_size}."
            )

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

    @property
    def train_batch_size(self) -> int:
        """Return the training batch size."""

        return OmegaConf.select(self.ds_config, "train_batch_size")

    @property
    def train_micro_batch_size_per_gpu(self) -> int:
        """Return the training batch size per GPU."""

        return OmegaConf.select(self.ds_config, "train_micro_batch_size_per_gpu")

    @property
    def is_local_main_process(self) -> bool:
        """Return whether the current process is the local main process."""

        return self.local_rank == 0

    @property
    def is_main_process(self) -> bool:
        """Return whether the current process is the global main process."""

        return self.rank == 0

    def to_dict(self, json_serialize: bool = False) -> Dict[str, Any]:
        """Convert attributes into a dictionary.

        Args:
            json_serialize: Whether to serialize non-compatible types into native types supported by JSON.

        Returns:
            Attributes encoded as a dictionary.

        """

        args = copy.deepcopy(asdict(self))

        if json_serialize:
            args["output_dir"] = str(self.output_dir)
            args["ds_config"] = OmegaConf.to_object(args["ds_config"])
            args["log_dir"] = str(args["log_dir"])

        return args
