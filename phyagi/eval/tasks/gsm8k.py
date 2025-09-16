# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import contextlib
import io
import re
import signal
from typing import Any, Dict, Optional, Union

import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from phyagi.eval.distributed_utils import (
    all_gather_list,
    all_reduce_dict,
    is_main_process,
)
from phyagi.eval.generation import generate
from phyagi.utils.file_utils import save_json_file


def _timeout_handler(signum: int, frame: Any) -> None:
    raise Exception()


def _parse_label(sentence: str) -> str:
    label_regex = re.compile(r"#### (\-?[0-9\.\,]+)")

    match = label_regex.search(sentence)
    if match:
        return match.group(1).strip().replace(",", "")

    return ""


def _validate_completion(completion: str, label: str) -> bool:
    completion_lines = completion.split("TA:")[1].strip().split("\n")
    completion_code = "\n".join(completion_lines[1:] if ":" in completion_lines[0] else completion_lines)

    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(2)

        try:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exec(
                    "import math\nfrom math import *\nimport numpy as np\nimport hashlib\n"
                    + completion_code
                    + "\n\n"
                    + "if type(result) == str:\n\tresult = result.replace(',', '')\n"
                    + f"assert(int(result) == {label})",
                    {},
                )
            signal.alarm(0)
            prediction = 1
        except Exception:
            prediction = 0
        finally:
            signal.alarm(0)

    except Exception:
        prediction = 0

    return prediction


class GSM8K:
    """GSM8K evaluation task.

    Reference:
        Training Verifiers to Solve Math Word Problems.
        https://arxiv.org/abs/2110.14168.

    """

    DATASET_PATH = "gsm8k"
    DATASET_NAME = "main"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "text": f"Student: {example['question']}\nTA:",
            "label": " " + example["answer"],
        }

    @staticmethod
    def run(
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        n_samples: int = 1,
        n_examples: Optional[int] = None,
        output_file_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        metric = load("accuracy")

        dataset = load_dataset(GSM8K.DATASET_PATH, name=GSM8K.DATASET_NAME)["test"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        generation_config = {
            "n_samples": n_samples,
            "stop_tokens": ["\n\n"],
            "do_sample": True if n_samples > 1 else False,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        }

        responses = generate(
            dataset,
            generation_engine="text_generation_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            example_generator_kwargs={"mapping_fn": GSM8K.mapping_fn},
            generation_config=generation_config,
            **kwargs,
        )

        outputs = []
        for r in responses:
            label = int(_parse_label(r["label"]))
            predictions = [_validate_completion(text, label) for text in r["responses"]]

            metric.add_batch(predictions=predictions, references=[1] * n_samples)
            outputs.append(
                {
                    "question": r["input"]["text"],
                    "answer": label,
                    "completion": [text.split(r["input"]["text"])[-1] for text in r["responses"]],
                    "passed": predictions,
                }
            )

        results = metric.compute()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

            outputs = all_gather_list(outputs)
            results = all_reduce_dict(results)

        if is_main_process():
            save_json_file(outputs, output_file_path) if output_file_path else None

        return results
