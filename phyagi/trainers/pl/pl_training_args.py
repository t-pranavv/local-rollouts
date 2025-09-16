# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import inspect
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import torch
from lightning.pytorch import Callback, seed_everything
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins import (
    CheckpointIO,
    ClusterEnvironment,
    LayerSync,
    Precision,
)
from lightning.pytorch.profilers import PyTorchProfiler

from phyagi.utils.file_utils import get_full_path
from phyagi.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PlStrategyArguments:
    """Define arguments used in the PyTorch Lightning strategy.

    Args:
        type: Strategy for distributed training (e.g., 'auto', 'ddp', 'dctp').
        data_parallel_size: Data parallel size for distributed training.
        context_parallel_size: Context parallel size for distributed training.
        tensor_parallel_size: Tensor parallel size for distributed training.
        activation_checkpointing: Whether to use activation checkpointing.
        fsdp_compile: Whether to compile the model in FSDP.
        fsdp_cpu_offload: Whether to offload computation to CPU in FSDP.
        tp_async: Whether to use asynchronous Tensor Parallelism.
        tp_sequence_parallel: Whether to use sequence parallelism in Tensor Parallelism.
            If ``True``, the input sequence must be evenly divisible by the tensor parallel size.
        tp_loss_parallel: Whether to use loss parallelism in Tensor Parallelism.

    """

    type: str = field(
        default="auto", metadata={"help": "Strategy for distributed training (e.g., 'auto', 'ddp', 'dctp')."}
    )

    data_parallel_size: int = field(default=1, metadata={"help": "Data parallel size for distributed training."})

    context_parallel_size: int = field(default=1, metadata={"help": "Context parallel size for distributed training."})

    tensor_parallel_size: int = field(default=1, metadata={"help": "Tensor parallel size for distributed training."})

    activation_checkpointing: bool = field(default=False, metadata={"help": "Whether to use activation checkpointing."})

    fsdp_compile: bool = field(default=False, metadata={"help": "Whether to compile the model in FSDP."})

    fsdp_cpu_offload: bool = field(default=False, metadata={"help": "Whether to offload computation to CPU in FSDP."})

    tp_async: bool = field(default=False, metadata={"help": "Whether to use asynchronous tensor parallelism."})

    tp_sequence_parallel: bool = field(
        default=False, metadata={"help": "Whether to use sequence parallelism in tensor parallelism."}
    )

    tp_loss_parallel: bool = field(
        default=False, metadata={"help": "Whether to use loss parallelism in tensor parallelism."}
    )

    def __post_init__(self) -> None:
        if self.data_parallel_size <= 1:
            if self.fsdp_compile:
                logger.warning("`fsdp_compile` is only supported when `data_parallel_size > 1`. Setting to False.")
                self.fsdp_compile = False
            if self.fsdp_cpu_offload:
                logger.warning("`fsdp_cpu_offload` is only supported when `data_parallel_size > 1`. Setting to False.")
                self.fsdp_cpu_offload = False

        if self.tensor_parallel_size <= 1:
            if self.tp_async:
                logger.warning("`tp_async` is only supported when `tensor_parallel_size > 1`. Setting to False.")
                self.tp_async = False

        if self.tp_async and not self.fsdp_compile:
            logger.warning("`tp_async` is only supported when `fsdp_compile` is enabled. Setting to False.")
            self.tp_async = False


@dataclass
class PlTrainerArguments:
    """Define arguments used in the PyTorch Lightning trainer.

    Args:
        accelerator: Accelerator to use (e.g., 'cpu', 'gpu', 'tpu').
        devices: Devices to use, e.g., 1 or 'auto'.
        num_nodes: Number of nodes for distributed training.
        precision: Precision setting (e.g., '16-mixed', 'bf16').
        logger: Logger(s) for tracking.
        callback: Callback(s) for extending training behavior.
        fast_dev_run: Run a single batch of training and testing.
        max_epochs: Maximum number of epochs.
        min_epochs: Minimum number of epochs.
        max_steps: Maximum number of training steps.
        min_steps: Minimum number of training steps.
        max_time: Maximum time for training.
        limit_train_batches: Fraction of training batches to use.
        limit_val_batches: Fraction of validation batches to use.
        limit_test_batches: Fraction of test batches to use.
        limit_predict_batches: Fraction of predict batches to use.
        overfit_batches: Fraction of data to overfit on.
        val_check_interval: Interval for validation checks.
        check_val_every_n_epoch: Frequency of validation per epoch.
        num_sanity_val_steps: Steps for sanity check validation.
        log_every_n_steps: Logging frequency in steps.
        enable_checkpointing: Enable model checkpointing.
        enable_progress_bar: Enable progress bar display.
        enable_model_summary: Enable model summary display.
        accumulate_grad_batches: Batch accumulation for gradient steps.
        gradient_clip_val: Value for gradient clipping.
        gradient_clip_algorithm: Algorithm for gradient clipping.
        deterministic: Use deterministic algorithms if True.
        benchmark: Set torch.backends.cudnn.benchmark.
        inference_mode: Use inference mode for evaluation.
        use_distributed_sampler: Use distributed sampler if True.
        profiler: Profiler type for bottleneck identification.
        detect_anomaly: Enable autograd anomaly detection.
        barebones: Disable features for raw speed analysis.
        plugins: Plugins for extending training behavior.
        sync_batchnorm: Synchronize batchnorm across processes.
        reload_dataloaders_every_n_epochs: Reload dataloaders every N epochs.
        default_root_dir: Root directory for checkpoints and logs.

    """

    accelerator: str = field(default="auto", metadata={"help": "Accelerator to use (e.g., 'cpu', 'gpu', 'tpu')."})

    devices: Union[List[int], str, int] = field(default="auto", metadata={"help": "Devices to use, e.g., 1 or 'auto'."})

    num_nodes: int = field(default=1, metadata={"help": "Number of nodes for distributed training."})

    precision: Union[int, str] = field(default=32, metadata={"help": "Precision setting (e.g., '16-mixed', 'bf16')."})

    logger: Optional[Union[bool, Logger, List[Logger]]] = field(
        default=None, metadata={"help": "Logger(s) for tracking."}
    )

    callbacks: Optional[Union[bool, Callback, List[Callback]]] = field(
        default=None, metadata={"help": "Callback(s) for extending training behavior."}
    )

    fast_dev_run: bool = field(default=False, metadata={"help": "Run a single batch of training and testing."})

    max_epochs: Optional[int] = field(default=None, metadata={"help": "Maximum number of epochs."})

    min_epochs: Optional[int] = field(default=None, metadata={"help": "Minimum number of epochs."})

    max_steps: int = field(default=-1, metadata={"help": "Maximum number of training steps."})

    min_steps: Optional[int] = field(default=None, metadata={"help": "Minimum number of training steps."})

    max_time: Optional[Union[str, timedelta, Dict[str, int]]] = field(
        default=None, metadata={"help": "Maximum time for training."}
    )

    limit_train_batches: Optional[Union[int, float]] = field(
        default=1.0, metadata={"help": "Fraction of training batches to use."}
    )

    limit_val_batches: Optional[Union[int, float]] = field(
        default=1.0, metadata={"help": "Fraction of validation batches to use."}
    )

    limit_test_batches: Optional[Union[int, float]] = field(
        default=1.0, metadata={"help": "Fraction of test batches to use."}
    )

    limit_predict_batches: Optional[Union[int, float]] = field(
        default=1.0, metadata={"help": "Fraction of predict batches to use."}
    )

    overfit_batches: Union[int, float] = field(default=0.0, metadata={"help": "Fraction of data to overfit on."})

    val_check_interval: Optional[Union[int, float]] = field(
        default=None, metadata={"help": "Interval for validation checks."}
    )

    check_val_every_n_epoch: Optional[int] = field(default=1, metadata={"help": "Frequency of validation per epoch."})

    num_sanity_val_steps: Optional[int] = field(default=2, metadata={"help": "Steps for sanity check validation."})

    log_every_n_steps: Optional[int] = field(default=50, metadata={"help": "Logging frequency in steps."})

    enable_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Enable model checkpointing."})

    enable_progress_bar: Optional[bool] = field(default=True, metadata={"help": "Enable progress bar display."})

    enable_model_summary: Optional[bool] = field(default=True, metadata={"help": "Enable model summary display."})

    accumulate_grad_batches: int = field(default=1, metadata={"help": "Batch accumulation for gradient steps."})

    gradient_clip_val: Optional[float] = field(default=None, metadata={"help": "Value for gradient clipping."})

    gradient_clip_algorithm: Optional[str] = field(
        default="norm", metadata={"help": "Algorithm for gradient clipping."}
    )

    deterministic: Optional[bool] = field(default=None, metadata={"help": "Use deterministic algorithms if True."})

    benchmark: Optional[bool] = field(default=None, metadata={"help": "Set torch.backends.cudnn.benchmark."})

    inference_mode: bool = field(default=True, metadata={"help": "Use inference mode for evaluation."})

    use_distributed_sampler: bool = field(default=True, metadata={"help": "Use distributed sampler if True."})

    profiler: Optional[Union[str, PyTorchProfiler]] = field(
        default=None, metadata={"help": "Profiler type for bottleneck identification."}
    )

    detect_anomaly: bool = field(default=False, metadata={"help": "Enable autograd anomaly detection."})

    barebones: bool = field(default=False, metadata={"help": "Disable features for raw speed analysis."})

    plugins: Optional[
        Union[
            Precision,
            ClusterEnvironment,
            CheckpointIO,
            LayerSync,
            List[Union[Precision, ClusterEnvironment, CheckpointIO, LayerSync]],
        ]
    ] = field(default=None, metadata={"help": "Plugins for extending training behavior."})

    sync_batchnorm: bool = field(default=False, metadata={"help": "Synchronize batchnorm across processes."})

    reload_dataloaders_every_n_epochs: int = field(default=0, metadata={"help": "Reload dataloaders every N epochs."})

    default_root_dir: Optional[str] = field(default=None, metadata={"help": "Root directory for checkpoints and logs."})

    def to_dict(self, json_serialize: bool = False) -> Dict[str, Any]:
        """Convert attributes into a dictionary.

        Args:
            json_serialize: Whether to serialize non-compatible types into native types supported by JSON.

        Returns:
            Attributes encoded as a dictionary.

        """

        def _serialize(obj: Union[Callable, TypeVar]) -> str:
            if inspect.isclass(obj) or inspect.ismethod(obj) or inspect.isfunction(obj):
                return obj.__name__
            if hasattr(obj, "__class__"):
                return obj.__class__.__name__
            return None

        args = copy.deepcopy(asdict(self))
        if json_serialize:
            args["logger"] = _serialize(args["logger"])
            args["callbacks"] = _serialize(args["callbacks"])
            args["profiler"] = _serialize(args["profiler"])
            args["plugins"] = _serialize(args["plugins"])

        return args


@dataclass
class PlLightningModuleArguments:
    """Define arguments used in the PyTorch Lightning module.

    Args:
        optimizer: Optimizer configuration to use for training.
        scheduler: Learning rate scheduler configuration to use for training.

    """

    optimizer: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Optimizer configuration to use for training."}
    )

    scheduler: Optional[Dict[str, Any]] = field(
        default=None, metadata={"help": "Learning rate scheduler configuration to use for training."}
    )


@dataclass
class PlTrainingArguments:
    """Define arguments used in the PyTorch Lightning training.

    Args:
        output_dir: Output folder where checkpoints and states will be written.
        strategy: Arguments for the PyTorch Lightning strategy.
        trainer: Arguments for the PyTorch Lightning trainer.
        lightning_module: Arguments for the PyTorch Lightning module.
        do_eval: Whether to run evaluation on the validation set.
        train_micro_batch_size_per_gpu: Batch size per GPU.
        save_steps: Number of steps between checkpoints.
        save_final_checkpoint: Whether to save last step checkpoint.
        seed: Random seed.
        dataloader_shuffle: Whether to shuffle the data loader (training).
        eval_dataloader_shuffle: Whether to shuffle the data loader (evaluation).
        dataloader_pin_memory: Whether to pin the data loader memory.
        dataloader_num_workers: Number of subprocesses to use for data loading.
        dataloader_prefetch_factor: Queue size for prefetch (per worker).
        log_dir: Directory to save logs. If not provided, will use ``output_dir``.
        mlflow: Whether to enable MLflow logging.
        wandb: Whether to enable Weights & Biases logging.
        tensorboard: Whether to enable TensorBoard logging.

    """

    output_dir: Union[str, Path] = field(
        metadata={"help": "Output folder where checkpoints and states will be written."}
    )

    strategy: PlStrategyArguments = field(default_factory=PlStrategyArguments)

    trainer: PlTrainerArguments = field(default_factory=PlTrainerArguments)

    lightning_module: PlLightningModuleArguments = field(default_factory=PlLightningModuleArguments)

    do_eval: bool = field(default=True, metadata={"help": "Whether to run evaluation on the validation set."})

    train_micro_batch_size_per_gpu: int = field(default=1, metadata={"help": "Batch size per GPU."})

    save_steps: int = field(
        default=500,
        metadata={"help": "Number of steps between checkpoints."},
    )

    save_final_checkpoint: bool = field(default=False, metadata={"help": "Whether to save last step checkpoint."})

    seed: int = field(default=42, metadata={"help": "Random seed."})

    dataloader_shuffle: bool = field(default=True, metadata={"help": "Whether to shuffle the data loader (training)."})

    eval_dataloader_shuffle: bool = field(
        default=True, metadata={"help": "Whether to shuffle the data loader (evaluation)."}
    )

    dataloader_pin_memory: bool = field(default=True, metadata={"help": "Whether to pin the data loader memory."})

    dataloader_num_workers: int = field(default=0, metadata={"help": "Number of subprocesses to use for data loading."})

    dataloader_prefetch_factor: int = field(default=None, metadata={"help": "Queue size for prefetch (per worker)."})

    log_dir: Union[str, Path] = field(default=None, metadata={"help": "Directory to save logs."})

    mlflow: bool = field(default=False, metadata={"help": "Whether to enable MLflow logging."})

    wandb: bool = field(default=False, metadata={"help": "Whether to enable Weights & Biases logging."})

    tensorboard: bool = field(default=False, metadata={"help": "Whether to enable TensorBoard logging."})

    def _init_pl_strategy_arguments(self) -> None:
        if isinstance(self.strategy, PlStrategyArguments):
            return
        self.strategy = PlStrategyArguments(**self.strategy)

    def _init_pl_trainer_arguments(self) -> None:
        if isinstance(self.trainer, PlTrainerArguments):
            return

        default_root_dir = self.trainer.pop("default_root_dir", None)
        if default_root_dir:
            logger.warning("`trainer.default_root_dir` has been supplied and overridden by `output_dir`.")

        self.trainer = PlTrainerArguments(
            **self.trainer,
            default_root_dir=self.output_dir,
        )

    def _init_pl_lightning_module_arguments(self) -> None:
        if isinstance(self.lightning_module, PlLightningModuleArguments):
            return
        self.lightning_module = PlLightningModuleArguments(**self.lightning_module)

    def __post_init__(self) -> None:
        torch.manual_seed(self.seed)
        seed_everything(self.seed)

        self.output_dir = get_full_path(self.output_dir, create_folder=True)
        self.log_dir = get_full_path(self.log_dir, create_folder=True) if self.log_dir is not None else self.output_dir

        self._init_pl_strategy_arguments()
        self._init_pl_trainer_arguments()
        self._init_pl_lightning_module_arguments()

    def to_dict(self, json_serialize: bool = False) -> Dict[str, Any]:
        """Convert attributes into a dictionary.

        Args:
            json_serialize: Whether to serialize non-compatible types into native types supported by JSON.

        Returns:
            Attributes encoded as a dictionary.

        """

        args = copy.deepcopy(asdict(self))

        args["trainer"] = self.trainer.to_dict(json_serialize=json_serialize)
        if json_serialize:
            args["output_dir"] = str(args["output_dir"])
            args["log_dir"] = str(args["log_dir"])

        return args
