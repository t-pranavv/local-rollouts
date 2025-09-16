# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
from pathlib import Path

import deepspeed

from phyagi.datasets.registry import get_data_collator, get_dataset
from phyagi.models.registry import get_model
from phyagi.trainers.registry import get_trainer, get_training_args
from phyagi.utils.config import load_config, save_config
from phyagi.utils.download_utils import download_dataset, download_tokenizer
from phyagi.version import get_package_information


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains/fine-tunes with DeepSpeed.")

    parser.add_argument(
        "config_file_path",
        type=str,
        nargs="*",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "-l",
        "--local_rank",
        type=int,
        default=-1,
        help="Rank of process passed by the DeepSpeed launcher.",
    )

    parser.add_argument(
        "-d",
        "--data_store",
        type=str,
        default="blobfuse",
        choices=["blobfuse", "azcopy"],
        help="Data downloading service.",
    )

    parser.add_argument(
        "-a",
        "--blob_account",
        type=str,
        default="phyagi",
        help="Blob storage account name.",
    )

    parser.add_argument(
        "-c",
        "--blob_container",
        type=str,
        default="data",
        help="Blob storage container name (should contain the data).",
    )

    parser.add_argument(
        "-r",
        "--local_data_root",
        type=str,
        default="/tmp/data",
        help="Path to store the downloaded data.",
    )

    parser.add_argument(
        "-t",
        "--tokenizer_dir",
        type=str,
        default=None,
        help="Path to the tokenizer directory.",
    )

    parser.add_argument(
        "-ck",
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to the checkpoint directory (for loading a pre-trained model).",
    )

    parser.add_argument(
        "-ckt",
        "--checkpoint_tag",
        type=str,
        default=None,
        help="Tag (number/step) of the desired checkpoint.",
    )

    args, extra_args = parser.parse_known_args()

    return args, extra_args


if __name__ == "__main__":
    args, extra_args = _parse_args()
    config = load_config(*args.config_file_path, extra_args)

    # Print information about the package (including branch and commit hash if available)
    print(get_package_information())

    # Ensure that the configuration file contains the necessary fields
    assert "output_dir" in config, "`output_dir` must be available in configuration."
    assert "dataset" in config, "`dataset` must be available in configuration."
    assert "model" in config, "`model` must be available in configuration."
    assert "training_args" in config, "`training_args` must be available in configuration."

    node_id = os.environ.get("USER_NODE_CODE_PACKAGE_INSTANCE_ID", None)
    if node_id is not None:
        print(f"USER_NODE_CODE_PACKAGE_INSTANCE_ID: {node_id}")

    # Load model from configuration
    model = get_model(**config["model"])
    model.train()

    # Load training arguments from configuration
    training_args = get_training_args(config["output_dir"], framework="ds", **config["training_args"])
    if args.checkpoint_dir is None:
        args.checkpoint_dir = training_args.output_dir

    # Save the input configuration only on the global main process
    if training_args.is_main_process:
        save_config(config, Path(training_args.output_dir) / "config.yaml")
    deepspeed.comm.barrier()

    # Load the training and evaluation (optional) datasets
    dataset_configs = config.get("dataset", [])
    dataset_concat = config.get("dataset_concat", "random")
    eval_dataset_concat = config.get("eval_dataset_concat", None)
    dataset_provider = config.get("dataset_provider", "lm")
    dataset_global_weight = config.get("dataset_global_weight", 1.0)

    # If `azcopy` is selected, download the datasets on the local main processes,
    # i.e., rank 0 on each node
    if args.data_store == "azcopy":
        data_root = config.get("data_root", None)
        assert data_root is not None, "`data_root` must be available in configuration when using `azcopy`."

        if training_args.is_local_main_process:
            download_dataset(
                dataset_configs, args.blob_account, args.blob_container, data_root, local_data_root=args.local_data_root
            )
        deepspeed.comm.barrier()

    # Download the tokenizer (if available)
    if args.tokenizer_dir is not None:
        download_tokenizer(
            args.blob_account, args.blob_container, args.tokenizer_dir, local_data_root=args.local_data_root
        )

    train_dataset, eval_dataset = get_dataset(
        dataset_configs,
        dataset_concat=dataset_concat,
        eval_dataset_concat=eval_dataset_concat,
        dataset_provider=dataset_provider,
        dataset_global_weight=dataset_global_weight,
    )

    # Load the data collator (optional)
    collator_config = config.get("dataset_collator", None)
    data_collator = get_data_collator(collator_config.pop("cls"), **collator_config) if collator_config else None

    # Initialize the trainer
    trainer = get_trainer(
        model,
        framework="ds",
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Run the training
    trainer.train(resume_from_checkpoint=args.checkpoint_dir, checkpoint_tag=args.checkpoint_tag)
