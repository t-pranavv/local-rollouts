# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from phyagi.trainers.pl.pl_training_args import (
    PlLightningModuleArguments,
    PlStrategyArguments,
    PlTrainerArguments,
    PlTrainingArguments,
)
from phyagi.utils.file_utils import get_full_path


def test_pl_strategy_arguments():
    args = PlStrategyArguments()
    assert args.type == "auto"
    assert args.data_parallel_size == 1
    assert args.context_parallel_size == 1
    assert args.tensor_parallel_size == 1
    assert args.fsdp_compile is False
    assert args.fsdp_cpu_offload is False
    assert args.tp_async is False
    assert args.tp_sequence_parallel is False
    assert args.tp_loss_parallel is False


@pytest.mark.parametrize("accelerator", ["cpu", "gpu", "tpu"])
def test_pl_trainer_arguments(accelerator):
    args = PlTrainerArguments(accelerator=accelerator)
    assert args.accelerator == accelerator
    assert args.devices == "auto"
    assert args.num_nodes == 1
    assert args.precision == 32
    assert args.logger is None
    assert args.callbacks is None
    assert args.fast_dev_run is False
    assert args.max_epochs is None
    assert args.min_epochs is None
    assert args.max_steps == -1
    assert args.min_steps is None
    assert args.max_time is None
    assert args.limit_train_batches == 1.0
    assert args.limit_val_batches == 1.0
    assert args.limit_test_batches == 1.0
    assert args.limit_predict_batches == 1.0
    assert args.overfit_batches == 0.0
    assert args.val_check_interval is None
    assert args.check_val_every_n_epoch == 1
    assert args.num_sanity_val_steps == 2
    assert args.log_every_n_steps == 50
    assert args.enable_checkpointing is True
    assert args.enable_progress_bar is True
    assert args.enable_model_summary is True
    assert args.accumulate_grad_batches == 1
    assert args.gradient_clip_val is None
    assert args.gradient_clip_algorithm == "norm"
    assert args.deterministic is None
    assert args.benchmark is None
    assert args.inference_mode is True
    assert args.use_distributed_sampler is True
    assert args.profiler is None
    assert args.detect_anomaly is False
    assert args.barebones is False
    assert args.plugins is None
    assert args.sync_batchnorm is False
    assert args.reload_dataloaders_every_n_epochs == 0
    assert args.default_root_dir is None


def test_pl_lightning_module_arguments():
    args = PlLightningModuleArguments()
    assert args.optimizer == {}
    assert args.scheduler is None


def test_pl_training_arguments():
    args = PlTrainingArguments(
        output_dir="output_dir",
        strategy=PlStrategyArguments(type="ddp"),
        trainer=PlTrainerArguments(accelerator="gpu"),
        lightning_module=PlLightningModuleArguments(),
        do_eval=True,
        train_micro_batch_size_per_gpu=8,
        save_steps=100,
        save_final_checkpoint=True,
        seed=42,
        dataloader_shuffle=True,
        eval_dataloader_shuffle=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        log_dir="logs",
        mlflow=False,
        wandb=False,
        tensorboard=False,
    )
    assert args.output_dir == get_full_path("output_dir")
    assert args.strategy.type == "ddp"
    assert args.trainer.accelerator == "gpu"
    assert args.lightning_module.optimizer == {}
    assert args.do_eval is True
    assert args.train_micro_batch_size_per_gpu == 8
    assert args.save_steps == 100
    assert args.save_final_checkpoint is True
    assert args.seed == 42
    assert args.dataloader_shuffle is True
    assert args.eval_dataloader_shuffle is True
    assert args.dataloader_pin_memory is True
    assert args.dataloader_num_workers == 4
    assert args.dataloader_prefetch_factor == 2
    assert args.log_dir == get_full_path("logs")
    assert args.mlflow is False
    assert args.wandb is False
    assert args.tensorboard is False
