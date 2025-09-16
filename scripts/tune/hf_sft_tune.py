# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import re
from pathlib import Path
from typing import Dict

from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import PreTrainedTokenizerBase
from trl import extract_prompt

from phyagi.datasets.rl.formatting_utils import (
    CHATML_CHAT_TEMPLATE,
    apply_chat_template,
)
from phyagi.models.registry import get_model, get_tokenizer
from phyagi.rl.tuners.registry import get_tuner, get_tuning_args
from phyagi.utils.config import load_config, save_config
from phyagi.version import get_package_information


def _prepare_dataset(
    dataset: Dataset, n_samples: int = None, system_message_format: str = "cot", system_message_prob: float = 0.25
) -> Dataset:
    assert isinstance(dataset, Dataset), "`dataset` must be an instance of `datasets.Dataset`."

    if n_samples:
        dataset = dataset.shuffle(seed=42).select(range(n_samples))

    # If the dataset has already been prepared, return it
    if "text" in dataset.column_names:
        return dataset

    # If the dataset has an implicit prompt, extract it
    if "prompt" not in dataset.column_names:
        dataset = dataset.map(extract_prompt, desc="Extracting prompts...")

    # If the dataset is a preference-based dataset, remove the negative examples
    # and rename the column "chosen" to "completion"
    if "chosen" in dataset.column_names:
        dataset = dataset.remove_columns("rejected").rename_column("chosen", "completion")

    assert "prompt" in dataset.column_names, "`prompt` must be available in the dataset."
    assert "completion" in dataset.column_names, "`completion` must be available in the dataset."

    # Apply chat template to the dataset and combine the prompt and completion columns
    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={
            "special_token_format": "chatml",
            "shuffle": False,
            "add_mask_tokens": False,
            "system_message_format": system_message_format,
            "system_message_prob": system_message_prob,
        },
        desc="Applying chat template...",
    )
    dataset = dataset.map(lambda x: {"text": x["prompt"] + x["completion"]}).remove_columns(
        ["prompt", "completion"], desc="Combining prompt and completion columns..."
    )

    print("Formatted data example:", dataset[0])

    return dataset


def _get_tokenized_length(messages: Dict[str, str], tokenizer: PreTrainedTokenizerBase) -> Dict[str, int]:
    input_text = messages["text"]
    input_ids = tokenizer(
        [input_text],
        add_special_tokens=False,
        return_tensors="np",
    )["input_ids"]

    return {"max_seq_len": input_ids[0].shape[0]}


def _filter_rows(row: Dict[str, str], tokenizer: PreTrainedTokenizerBase, max_length: int) -> bool:
    return _get_tokenized_length(row, tokenizer=tokenizer)["max_seq_len"] <= max_length


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tunes with Hugging Face.")

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

    parser.add_argument(
        "-n",
        "--n_train_samples",
        type=int,
        default=None,
        help="Number of training data samples to use, if None will use entire data.",
    )

    parser.add_argument(
        "-fl",
        "--filter_length",
        action="store_true",
        help="Filter data samples that go beyond the tuner's max_length argument.",
    )

    parser.add_argument(
        "-smf",
        "--system_message_format",
        type=str,
        default="cot_final",
        help="select system message from random or reasoning cot. options are ['random', 'cot_final', 'cot_tools']",
    )

    parser.add_argument(
        "-smp",
        "--system_message_prob",
        type=float,
        default=1.0,
        help="Probability of using system message in the conversation.",
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
    assert "tuning_args" in config, "`tuning_args` must be available in configuration."

    # Load model and tokenizer from configuration
    # If tokenizer is not available, it will be defaulted to the model
    model = get_model(**config["model"])

    tokenizer_config = config.get("tokenizer", {})
    if tokenizer_config.get("pretrained_tokenizer_name_or_path", None) is None:
        tokenizer_config["pretrained_tokenizer_name_or_path"] = model.config.name_or_path

    tokenizer = get_tokenizer(**tokenizer_config)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHATML_CHAT_TEMPLATE

    # Load tuning arguments from configuration
    tuning_args = get_tuning_args(config["output_dir"], framework="hf", task="sft", **config["tuning_args"])
    if args.checkpoint_dir is None:
        args.checkpoint_dir = tuning_args.output_dir

    # Load PEFT configuration (optional)
    peft_config = config.get("peft_config", None)
    if peft_config is not None:
        peft_config = LoraConfig(**peft_config)

    # Load and prepare the training and evaluation (optional) datasets
    train_dataset = load_dataset(**config["dataset"], split="train")
    train_dataset = _prepare_dataset(
        train_dataset,
        n_samples=args.n_train_samples,
        system_message_format=args.system_message_format,
        system_message_prob=args.system_message_prob,
    )
    if args.filter_length:
        num_samples_before_filtering = len(train_dataset)
        train_dataset = train_dataset.filter(
            _filter_rows,
            fn_kwargs={"tokenizer": tokenizer, "max_length": tuning_args.max_length},
            num_proc=16,
            desc=f"Filtering length >= {tuning_args.max_length}...",
        )

        print(f"Filtered {num_samples_before_filtering - len(train_dataset)} out of {len(train_dataset)} samples.")

    # Save the input configuration
    os.makedirs(tuning_args.output_dir, exist_ok=True)
    save_config(config, Path(tuning_args.output_dir) / "config.yaml")

    # Initialize the tuner
    tuner = get_tuner(
        framework="hf",
        task="sft",
        model=model,
        tuning_args=tuning_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Run the tuning
    tuner.train(
        resume_from_checkpoint=args.checkpoint_dir if re.match(r"checkpoint-\d+", args.checkpoint_dir) else False
    )
