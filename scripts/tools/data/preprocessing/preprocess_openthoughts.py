# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset

SYSTEM_MESSAGE = "Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <|dummy_86|> {Thought section} <|dummy_87|> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:"


def _pre_process(sample: Dict[str, Any]) -> Dict[str, Any]:
    new_messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

    for msg in sample["messages"]:
        processed = {"role": msg["from"], "content": msg["value"]}
        if processed["role"] == "assistant":
            processed["content"] = processed["content"].replace("<|begin_of_thought|>\n\n", "<|dummy_86|>")
            processed["content"] = processed["content"].replace(
                "\n\n<|end_of_thought|>\n\n<|begin_of_solution|>\n\n", "<|dummy_87|>"
            )
            processed["content"] = processed["content"].replace("\n\n<|end_of_solution|>", "")
        new_messages.append(processed)

    return {"messages": new_messages}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-process OpenThoughts dataset for RLHF.")

    parser.add_argument("dataset_name", type=str, help="Dataset name.")

    parser.add_argument("output_file", type=Path, help="Path to the output JSONL file.")

    parser.add_argument(
        "-n",
        "--num-proc",
        type=int,
        default=16,
        help="Number of processes to use for dataset.map().",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name)

    train_dataset = dataset["train"].remove_columns(["system"]).rename_column("conversations", "messages")
    processed_dataset = train_dataset.map(_pre_process, num_proc=args.num_proc, desc="Pre-processing dataset...")

    print(f"Writing output: {args.output_file}")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as fout:
        for sample in processed_dataset:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"{len(processed_dataset)} samples written.")
