# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

from datasets import load_dataset

SYSTEM_MESSAGE = (
    "Your role as an assistant involves thoroughly exploring questions through a systematic thinking process "
    "before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle "
    "of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop a "
    "well-considered thinking process. Please structure your response into two main sections: Thought and Solution "
    "using the specified format: <|dummy_86|> {Thought section} <|dummy_87|> {Solution section}. In the "
    "Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as "
    "analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current "
    "steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, "
    "explorations, and reflections from the Thought section, systematically present the final solution that you deem "
    "correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to "
    "reach the conclusion. Now, try to solve the following question through the above guidelines:"
)

START_THINK = "<think>"
END_THINK = "</think>"


def _process_string(s: str) -> str:
    prefix = START_THINK
    if s.startswith(prefix):
        body = s[len(prefix) :].lstrip()
    else:
        body = s

    tokens = body.split()

    for i in range(min(3, len(tokens))):
        token = tokens[i]
        if re.search(r"(alright|okay)", token, re.IGNORECASE):
            tokens[i] = (
                token[token.lower().find("alright") : token.lower().find("alright")]
                if "alright" in token.lower()
                else token[token.lower().find("okay") :]
            )
            new_body = " ".join(tokens[i:])
            return prefix + new_body

    if len(tokens) >= 2:
        allowed_single = {"let's", "let", "i", "so"}

        second = tokens[1].lower()
        if second in allowed_single or (second == "the" and len(tokens) >= 3 and tokens[2].lower() == "user"):
            return prefix + " ".join(tokens[1:])

    return s


def _replace_unique_substring(original: str, substring: str, replacement: str = "") -> str:
    if original.count(substring) != 1:
        raise ValueError(f"{substring} must occur exactly once.")
    return original.replace(substring, replacement, 1)


def _preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
    new_messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    prompt = example["prompt"][0]
    new_messages.append({"role": "user", "content": prompt})

    response = _process_string(example["llm_response"][0])
    try:
        response = _replace_unique_substring(response, START_THINK, "<|dummy_86|>")
        response = _replace_unique_substring(response, END_THINK, "<|dummy_87|>")
    except ValueError as e:
        print(f"Skipping sample: {e}")
        return {"messages": []}

    new_messages.append({"role": "assistant", "content": response})

    return {"messages": [new_messages]}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-process a Hugging Face dataset for RLHF.")

    parser.add_argument("dataset_name", type=str, help="Hugging Face dataset identifier.")

    parser.add_argument("output_file", type=Path, help="Path to the output JSONL file.")

    parser.add_argument("-ms", "--min-score", type=float, default=0.7, help="Minimum score threshold for filtering.")

    parser.add_argument("-s", "--split", type=str, default="train", help="Dataset split to load.")

    parser.add_argument("-n", "--num-proc", type=int, default=16, help="Number of processes for map/filter operations.")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    print(f"Loading dataset: {args.dataset_name} (split={args.split})")
    dataset = load_dataset(args.dataset_name, split=args.split)

    dataset = dataset.filter(
        lambda ex, threshold=args.min_score: ex["score"] is not None and ex["score"] >= threshold,
        num_proc=args.num_proc,
        desc=f"Filtering samples with score >= {args.min_score}...",
    )
    processed_dataset = dataset.map(
        _preprocess,
        num_proc=args.num_proc,
        batched=True,
        batch_size=1,
        remove_columns=dataset.column_names,
        desc="Pre-processing dataset...",
    )

    print(f"Writing output: {args.output_file}")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as fout:
        for sample in processed_dataset:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"{len(processed_dataset)} samples written.")
