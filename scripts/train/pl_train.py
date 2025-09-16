# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
from pathlib import Path

from phyagi.datasets.registry import get_data_collator, get_dataset
from phyagi.models.registry import get_model
from phyagi.trainers.registry import get_trainer, get_training_args
from phyagi.utils.config import load_config, save_config
from phyagi.version import get_package_information


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trains/fine-tunes with PyTorch Lightning.")

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
        help="Rank of process passed by the PyTorch launcher.",
    )

    parser.add_argument(
        "-ck",
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to the checkpoint directory (for loading a pre-trained model).",
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

    # Load model from configuration
    model = get_model(**config["model"])
    model.train()

    # Load training arguments from configuration
    training_args = get_training_args(config["output_dir"], framework="pl", **config["training_args"])
    if args.checkpoint_dir is None:
        args.checkpoint_dir = training_args.output_dir

    # Load the training and evaluation (optional) datasets
    dataset_configs = config.get("dataset", [])
    dataset_concat = config.get("dataset_concat", "random")
    eval_dataset_concat = config.get("eval_dataset_concat", None)
    dataset_provider = config.get("dataset_provider", "lm")
    dataset_global_weight = config.get("dataset_global_weight", 1.0)

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
        framework="pl",
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # In `PlTrainer`, we only have access to the ranks after the trainer is initialized
    if trainer.is_global_zero:
        # Save the input configuration
        os.makedirs(training_args.output_dir, exist_ok=True)
        save_config(config, Path(training_args.output_dir) / "config.yaml")

    # Run the training
    trainer.train()
