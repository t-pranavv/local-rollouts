# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import subprocess
from pathlib import Path

from phyagi.datasets.registry import get_data_collator, get_dataset
from phyagi.models.registry import get_tokenizer
from phyagi.rl.rewards.registry import get_reward
from phyagi.rl.tuners.registry import get_tuner, get_tuning_args
from phyagi.utils.config import load_config, save_config
from phyagi.version import get_package_information


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tunes a model with ISFT and Ray.")

    parser.add_argument(
        "config_file_path",
        type=str,
        nargs="*",
        help="Path to the YAML configuration file.",
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
    assert "tokenizer" in config, "`tokenizer` must be available in configuration."
    assert "dataset" in config, "`dataset` must be available in configuration."
    assert "rewards" in config, "`rewards` must be available in configuration."
    assert "tuning_args" in config, "`tuning_args` must be available in configuration."

    # Load tuning arguments from configuration
    tuning_args = get_tuning_args(config["output_dir"], framework="ray", task="isft", **config["tuning_args"])

    # Load the tokenizer and rewards from configuration
    tokenizer = get_tokenizer(**config["tokenizer"])
    rewards = get_reward(config["rewards"])

    # Load the training and evaluation (optional) datasets
    dataset_configs = config.get("dataset", [])
    train_dataset, eval_dataset = get_dataset(
        dataset_configs,
        dataset_concat="random_chat",
        eval_dataset_concat="random_chat",
        dataset_provider="chat",
    )

    # `get_dataset` creates a dictionary of evaluation datasets, but ISFT only
    # supports a single evaluation dataset
    if isinstance(eval_dataset, dict):
        eval_dataset = list(eval_dataset.values())[0]

    # Load the data collator
    data_collator = get_data_collator(data_collator_type="chat", reward_names=list(rewards.keys()))

    # Save the input configuration
    os.makedirs(tuning_args.output_dir, exist_ok=True)
    save_config(config, Path(tuning_args.output_dir) / "config.yaml")

    # Initialize the tuner
    tuner = get_tuner(
        framework="ray",
        task="isft",
        tuning_args=tuning_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        rewards=rewards,
    )

    # Run the tuning
    tuner.train(resume_from_checkpoint=args.checkpoint_dir, checkpoint_tag=args.checkpoint_tag)
    subprocess.run(["ray", "stop", "--force"], check=True)
