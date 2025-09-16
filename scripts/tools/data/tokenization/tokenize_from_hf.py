# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from pathlib import Path

from phyagi.datasets.train.lm.lm_dataset_provider import LMDatasetProvider


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize a Hugging Face dataset.")

    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for the raw/tokenized dataset.",
    )

    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default="microsoft/phi-2",
        help="Name of the tokenizer to use.",
    )

    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        required=True,
        help="Path of the Hugging Face dataset.",
    )

    parser.add_argument(
        "-dn",
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the Hugging Face dataset configuration.",
    )

    parser.add_argument(
        "-dr",
        "--revision",
        type=str,
        default=None,
        help="Revision of the Hugging Face dataset.",
    )

    parser.add_argument(
        "--save_raw",
        action="store_true",
        help="Whether to save the raw dataset.",
    )

    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=64,
        help="Number of workers to use for the dataset provider.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_args()

    dataset_provider = LMDatasetProvider.from_hub(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        revision=args.revision,
        tokenizer=args.tokenizer,
        validation_split=0.1,
        shuffle=True,
        use_eos_token=True,
        num_workers=args.num_workers,
        cache_dir=Path(args.output_dir) / args.dataset_path,
        raw_dir=Path(args.output_dir) / args.dataset_path / "raw" if args.save_raw else None,
    )
