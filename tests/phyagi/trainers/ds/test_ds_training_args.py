# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import shutil
import tempfile

import pytest

from phyagi.trainers.ds.ds_training_args import DsTrainingArguments
from phyagi.utils.file_utils import get_full_path


@pytest.mark.is_mpi
def test_ds_training_arguments():
    args = DsTrainingArguments("output")
    assert args.output_dir == get_full_path("output")
    assert args.do_eval is True
    assert args.do_final_eval is False
    assert args.train_batch_size_init_rampup == 0
    assert args.train_batch_size_per_rampup == 0
    assert args.rampup_steps == 0
    assert args.num_train_epochs == 1
    assert args.max_steps == -1
    assert args.logging_steps == 10
    assert args.save_steps == 500
    assert args.save_final_checkpoint is False
    assert args.eval_steps == 500
    assert args.eval_max_steps is None
    assert args.seed == 42
    assert args.pipe_parallel_size == 1
    assert args.pipe_parallel_partition_method == "parameters"
    assert args.pipe_parallel_activation_checkpoint_steps == 0
    assert args.tensor_parallel_size == 1
    assert args.context_parallel_size == 1
    assert args.batch_tracker is False
    assert args.batch_tracker_save_steps == 10
    assert args.dataloader_shuffle is True
    assert args.eval_dataloader_shuffle is True
    assert args.dataloader_pin_memory is True
    assert args.dataloader_num_workers == 0
    assert args.dataloader_prefetch_factor is None
    assert args.load_checkpoint_num_tries == 1
    assert args.backend is None
    assert args.timeout == 1800
    assert args.log_dir == get_full_path("output")
    assert args.mlflow is False
    assert args.wandb is False
    assert args.wandb_api_key is None
    assert args.wandb_host is None
    assert args.tensorboard is False

    custom_args = DsTrainingArguments(
        output_dir="custom_output",
        do_eval=False,
        do_final_eval=True,
        train_batch_size_init_rampup=5,
        train_batch_size_per_rampup=5,
        rampup_steps=10,
        num_train_epochs=2,
        max_steps=2,
        logging_steps=20,
        save_steps=1000,
        save_final_checkpoint=True,
        seed=123,
        eval_steps=1000,
        eval_max_steps=10,
        pipe_parallel_size=1,
        pipe_parallel_partition_method="custom_partition",
        pipe_parallel_activation_checkpoint_steps=1,
        tensor_parallel_size=1,
        context_parallel_size=1,
        batch_tracker=True,
        batch_tracker_save_steps=100,
        dataloader_shuffle=False,
        eval_dataloader_shuffle=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=1,
        load_checkpoint_num_tries=2,
        backend="nccl",
        timeout=900,
        log_dir="custom_output",
        mlflow=False,
        wandb=False,
        wandb_api_key="custom_api_key",
        wandb_host="custom_host",
        tensorboard=False,
    )
    assert custom_args.output_dir == get_full_path("custom_output")
    assert custom_args.do_eval is False
    assert custom_args.do_final_eval is True
    assert custom_args.train_batch_size_init_rampup == 5
    assert custom_args.train_batch_size_per_rampup == 5
    assert custom_args.rampup_steps == 10
    assert custom_args.num_train_epochs == 2
    assert custom_args.max_steps == 2
    assert custom_args.logging_steps == 20
    assert custom_args.save_steps == 1000
    assert custom_args.save_final_checkpoint is True
    assert custom_args.seed == 123
    assert custom_args.eval_steps == 1000
    assert custom_args.eval_max_steps == 10
    assert custom_args.pipe_parallel_size == 1
    assert custom_args.pipe_parallel_partition_method == "custom_partition"
    assert custom_args.pipe_parallel_activation_checkpoint_steps == 1
    assert custom_args.tensor_parallel_size == 1
    assert custom_args.context_parallel_size == 1
    assert custom_args.batch_tracker is True
    assert custom_args.batch_tracker_save_steps == 100
    assert custom_args.dataloader_shuffle is False
    assert custom_args.eval_dataloader_shuffle is False
    assert custom_args.dataloader_pin_memory is False
    assert custom_args.dataloader_num_workers == 4
    assert custom_args.dataloader_prefetch_factor == 1
    assert custom_args.load_checkpoint_num_tries == 2
    assert custom_args.backend == "nccl"
    assert custom_args.timeout == 900
    assert custom_args.log_dir == get_full_path("custom_output")
    assert custom_args.mlflow is False
    assert custom_args.wandb is False
    assert custom_args.wandb_api_key == "custom_api_key"
    assert custom_args.wandb_host == "custom_host"
    assert custom_args.tensorboard is False

    custom_ds_config = {"train_batch_size": 64, "train_micro_batch_size_per_gpu": 8}
    args_with_ds_config = DsTrainingArguments(output_dir="output", ds_config=custom_ds_config)
    assert args_with_ds_config.ds_config["train_batch_size"] == 64
    assert args_with_ds_config.ds_config["train_micro_batch_size_per_gpu"] == 8

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as tmpfile:
        json.dump(custom_ds_config, tmpfile, indent=4)
        tmpfile.flush()

        args_with_ds_config_file = DsTrainingArguments(output_dir="output", ds_config=tmpfile.name)
        assert args_with_ds_config_file.ds_config["train_batch_size"] == 64
        assert args_with_ds_config_file.ds_config["train_micro_batch_size_per_gpu"] == 8

    shutil.rmtree("output", ignore_errors=True)
    shutil.rmtree("custom_output", ignore_errors=True)
