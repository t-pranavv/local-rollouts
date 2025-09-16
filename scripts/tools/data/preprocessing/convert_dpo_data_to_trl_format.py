# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

START_TOKEN = "<|im_start|>"
SEP_TOKEN = "<|im_sep|>"
END_TOKEN = "<|im_end|>"


def _parse_prefix_text(prefix_text: str) -> List[Dict[str, str]]:
    messages = []

    for segment in prefix_text.split(START_TOKEN):
        if not segment.strip() or SEP_TOKEN not in segment:
            continue

        role, content = segment.split(SEP_TOKEN, 1)

        if END_TOKEN in content:
            content = content.split(END_TOKEN, 1)[0]

        messages.append({"role": role.strip(), "content": content})

    return messages


def _convert_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    prefix = raw.get("prefix_text", "")
    chosen_text = raw.get("text", "")
    rejected_text = raw.get("text2", "")

    messages = _parse_prefix_text(prefix)

    assistant_prefix = ""
    if messages and messages[-1]["role"] == "assistant":
        assistant_prefix = messages.pop()["content"]

    chosen_content = assistant_prefix + chosen_text
    rejected_content = assistant_prefix + rejected_text

    return {
        "prompt": messages,
        "chosen": [{"role": "assistant", "content": chosen_content}],
        "rejected": [{"role": "assistant", "content": rejected_content}],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert prefix-formatted JSONL files to prompt/chosen/rejected layout."
    )

    parser.add_argument("input_file", type=Path, help="Input .jsonl file.")

    parser.add_argument("output_file", type=Path, help="Output .jsonl file.")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    n_inputs = n_outputs = 0
    with args.input_file.open("r", encoding="utf-8") as fin, args.output_file.open("w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, 1):
            n_inputs += 1
            line = line.rstrip("\n")

            if not line:
                continue

            try:
                raw: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                print("Line %d: JSON decode error - %s", line_num, exc)
                continue

            converted = _convert_record(raw)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")

            n_outputs += 1
