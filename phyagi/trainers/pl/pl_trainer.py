# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import (
    CSVLogger,
    MLFlowLogger,
    TensorBoardLogger,
    WandbLogger,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, IterableDataset

from phyagi.datasets.concat_dataset import WeightedConcatIterableDataset
from phyagi.trainers.pl.pl_callbacks import MetricLogCallback, OptimizerLogCallback
from phyagi.trainers.pl.pl_lightning_module import TrainingLightningModule
from phyagi.trainers.pl.pl_progress_bars import TQDMStepProgressBar
from phyagi.trainers.pl.pl_strategies import DataContextTensorParallelStrategy
from phyagi.trainers.pl.pl_training_args import PlTrainingArguments


class PlTrainer(Trainer):
    """PyTorch Lightning trainer."""

    def __init__(
        self,
        model: torch.nn.Module,
        args: Optional[PlTrainingArguments] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        optimizers: Optional[Tuple[Optimizer, LRScheduler]] = None,
        **kwargs,
    ) -> None:
        """Initialize the PyTorch Lightning trainer.

        Args:
            model: Model to be trained or evaluated.
            args: PyTorch Lightning training arguments.
                If set to ``None``, will use a default instance of :class:`phyagi.trainers.pl.pl_training_args.PlTrainingArguments`.
            data_collator: Collate function used for creating a batch from ``train_dataset`` and ``eval_dataset``.
            train_dataset: Dataset used for training.
                If set to ``None``, :meth:`train` will not be available.
            eval_dataset: Dataset used for evaluation.
                If set to ``None``, will not perform evaluation.
            optimizers: Tuple of ``(optimizer, lr_scheduler)`` to be used for training.
                If set to ``None``, will use the optimizer and scheduler defined in DeepSpeed configuration.

        """

        args = args or PlTrainingArguments("tmp")
        if not isinstance(args, PlTrainingArguments):
            raise TypeError(f"`args` must be an instance of PlTrainingArguments, but got '{type(args)}'.")
        self._args = args

        self._data_collator = data_collator
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._args.do_eval = self._args.do_eval if self._eval_dataset is not None else False

        optimizers = optimizers or (None, None)
        if not (isinstance(optimizers, tuple) and len(optimizers) == 2):
            raise ValueError(f"`optimizers` must be a tuple of length 2, but got {optimizers}.")
        optimizer, scheduler = optimizers

        # Prepare the training strategy
        strategy = self._args.strategy.type
        if strategy == DataContextTensorParallelStrategy.STRATEGY_TYPE:
            strategy = DataContextTensorParallelStrategy(
                data_parallel_size=self._args.strategy.data_parallel_size,
                context_parallel_size=self._args.strategy.context_parallel_size,
                tensor_parallel_size=self._args.strategy.tensor_parallel_size,
                cpu_offload=self._args.strategy.fsdp_cpu_offload,
            )
        
        # Prepare the model
        self.pl_module = TrainingLightningModule(
            model,
            self._args.lightning_module,
            self._args.strategy,
            optimizer,
            scheduler,
        )

        # Prepare the trainer arguments
        trainer_kwargs = self._args.trainer.to_dict()

        if trainer_kwargs["callbacks"] is None:
            model_checkpoint = ModelCheckpoint(
                dirpath=self._args.output_dir,
                filename="{step}",
                auto_insert_metric_name=False,
                every_n_train_steps=self._args.save_steps if self._args.save_steps > 0 else None,
                save_last=self._args.save_final_checkpoint,
                save_top_k=-1,
                save_on_train_epoch_end=False,
            )
            model_checkpoint.CHECKPOINT_EQUALS_CHAR = ""
            model_checkpoint.FILE_EXTENSION = ""

            metric_log = MetricLogCallback(
                log_every_n_steps=self._args.trainer.log_every_n_steps,
                enable_loss_parallel=self._args.strategy.tp_loss_parallel,
            )
            optimizer_log = OptimizerLogCallback(log_every_n_steps=self._args.trainer.log_every_n_steps)
            step_progress_bar = TQDMStepProgressBar()

            trainer_kwargs["callbacks"] = [model_checkpoint, metric_log, optimizer_log, step_progress_bar]

        if trainer_kwargs["logger"] is None:
            config = {
                "model_config": self.pl_module.model_config.to_dict() if self.pl_module.model_config else None,
                "training_args": self._args.to_dict(json_serialize=True),
            }

            trainer_kwargs["logger"] = [CSVLogger(save_dir=self._args.log_dir, name=None, version="")]
            if self._args.mlflow:
                trainer_kwargs["logger"].append(MLFlowLogger(tags=config, save_dir=self._args.log_dir))
            if self._args.wandb:
                # `WandbLogger` does not support passing `key` and `host` to initialize the connection
                wandb.login(
                    key=os.environ.get("WANDB_API_KEY", None), host=os.environ.get("WANDB_HOST", None), timeout=0
                )
                trainer_kwargs["logger"].append(WandbLogger(config=config, save_dir=self._args.log_dir))
            if self._args.tensorboard:
                trainer_kwargs["logger"].append(TensorBoardLogger(save_dir=self._args.log_dir))

        super().__init__(strategy=strategy, **trainer_kwargs)

    @property
    def _data_parallel_size(self) -> int:
        """Return the data parallel size."""

        return getattr(self.strategy, "_data_parallel_size", self.world_size)

    @property
    def _context_parallel_size(self) -> int:
        """Return the context parallel size."""

        return getattr(self.strategy, "_context_parallel_size", 1)

    @property
    def _tensor_parallel_size(self) -> int:
        """Return the tensor parallel size."""

        return getattr(self.strategy, "_tensor_parallel_size", 1)

    @property
    def train_batch_size(self) -> int:
        """Return the training batch size."""

        return self._args.train_micro_batch_size_per_gpu * self._data_parallel_size * self.accumulate_grad_batches

    @property
    def seq_len(self) -> int:
        """Return the sequence length."""

        return getattr(self._train_dataset, "seq_len", 1) if self._train_dataset is not None else 1

    def save_checkpoint(
        self, filepath: Union[str, Path], weights_only: bool = False, storage_options: Optional[Any] = None
    ) -> None:
        super().save_checkpoint(filepath, weights_only, storage_options)

        if self.is_global_zero and self.pl_module.model_config:
            save_path = Path(filepath)
            if save_path.is_dir():
                self.pl_module.model_config.save_pretrained(save_path)
            else:
                self.pl_module.model_config.save_pretrained(save_path.parent)

    def get_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Get a data loader for a given dataset.

        When using with supported strategy, e.g., `auto`, `ddp` and `mp`, the data loader
        will be created with the appropriate topology and will be automatically distributed across devices.

        Args:
            dataset: Dataset to be used.
            shuffle: Whether to shuffle the dataset.

        Returns:
            Data loader.

        """

        if isinstance(dataset, WeightedConcatIterableDataset):
            raise TypeError("`dataset` as an instance of WeightedConcatIterableDataset is not supported yet.")

        return DataLoader(
            dataset,
            batch_size=self._args.train_micro_batch_size_per_gpu,
            shuffle=shuffle,
            num_workers=self._args.dataloader_num_workers,
            pin_memory=self._args.dataloader_pin_memory if not isinstance(dataset, IterableDataset) else True,
            drop_last=True,
            collate_fn=self._data_collator,
            prefetch_factor=self._args.dataloader_prefetch_factor,
        )

    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """Train a model.

        Args:
            resume_from_checkpoint: Resume training from a specific checkpoint.
                If set to ``None``, training will start from scratch.
                If different than ``None``, training will resume from the checkpoint.

        """

        if self._train_dataset is None:
            raise ValueError("`train_dataset` must be defined, but got None.")
        train_dataloader = self.get_dataloader(self._train_dataset, shuffle=self._args.dataloader_shuffle)

        val_dataloader = None
        if self._args.do_eval:
            val_dataloader = self.get_dataloader(self._eval_dataset, shuffle=self._args.eval_dataloader_shuffle)

        self.fit(
            self.pl_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=resume_from_checkpoint,
        )

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> None:
        """Evaluate a model.

        Args:
            eval_dataset: Evaluation dataset.

        """

        if eval_dataset is None:
            raise ValueError("`eval_dataset` must be defined, but got None.")
        eval_dataloader = self.get_dataloader(eval_dataset, shuffle=self._args.eval_dataloader_shuffle)

        self.validate(model=self.pl_module, dataloaders=eval_dataloader)

    def predict(self) -> None:
        """Predict with a model."""

        raise NotImplementedError
