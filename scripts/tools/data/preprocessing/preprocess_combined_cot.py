# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import datasets

START_SOLUTION = "<|begin_of_solution|>"
END_SOLUTION = "<|end_of_solution|>"
START_THOUGHT = "<|begin_of_thought|>"
END_THOUGHT = "<|end_of_thought|>"
DUMMY_86 = "<|dummy_86|>"
DUMMY_87 = "<|dummy_87|>"


def _replace_unique_substring(original: str, substring: str, replacement: str = "") -> str:
    if original.count(substring) != 1:
        raise ValueError(f"{substring} must occur exactly once.")
    return original.replace(substring, replacement, 1)


def _pre_process(sample: Dict[str, Any]) -> Dict[str, Any]:
    sys_msg = sample["system"][0]

    try:
        sys_msg = _replace_unique_substring(sys_msg, f"{START_SOLUTION} ")
        sys_msg = _replace_unique_substring(sys_msg, f" {END_SOLUTION}")
        sys_msg = _replace_unique_substring(sys_msg, START_THOUGHT, DUMMY_86)
        sys_msg = _replace_unique_substring(sys_msg, END_THOUGHT, DUMMY_87)
    except ValueError as e:
        print(f"System message skipped: {e}")
        return {"messages": []}

    new_messages = [{"role": "system", "content": sys_msg}]

    for msg in sample["conversations"][0]:
        processed = {"role": msg["from"], "content": msg["value"]}
        if processed["role"] == "assistant":
            try:
                processed["content"] = _replace_unique_substring(processed["content"], START_SOLUTION)
                processed["content"] = _replace_unique_substring(processed["content"], END_SOLUTION)
                processed["content"] = _replace_unique_substring(processed["content"], START_THOUGHT, DUMMY_86)
                processed["content"] = _replace_unique_substring(processed["content"], END_THOUGHT, DUMMY_87)
            except ValueError as e:
                print(f"Assistant message skipped: {e}")
                return {"messages": []}
        new_messages.append(processed)

    return {"messages": [new_messages]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess OpenThoughts dataset for RLHF.")

    parser.add_argument("input_dir", type=Path, help="Directory containing the dataset on disk.")

    parser.add_argument("output_file", type=Path, help="Path to the JSONL file to write.")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print(f"Loading dataset: {args.input_dir}")
    dataset = datasets.load_from_disk(str(args.input_dir))

    processed_dataset = dataset.map(
        _pre_process,
        num_proc=16,
        remove_columns=["system", "conversations"],
        batched=True,
        batch_size=1,
        desc="Pre-processing dataset...",
    )

    print(f"Writing output: {args.output_file}")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        for sample in processed_dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"{len(processed_dataset)} samples written.")
