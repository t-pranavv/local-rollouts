# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import multiprocessing
import os
import re
from collections import Counter
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


def _parse_label_from_solution(text: str) -> str:
    label_regex = re.compile(r"\\boxed{([a-z0-9\.\,\\{}$]+)}")
    match = label_regex.search(text)

    return match.group(1).strip() if match else ""


class MATH:
    """MATH evaluation task.

    Reference:
        Measuring Mathematical Problem Solving With the MATH Dataset.
        https://arxiv.org/abs/2103.03874.

    """

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    DATASET_PATH = "competition_math"
    DATASET_NAME = None

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "level": example["level"],
            "type": example["type"],
            "text": example["problem"],
            "label": _parse_label_from_solution(example["solution"]),
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
        # Avoid error when multiprocessing from outside
        multiprocessing.set_start_method("fork", force=True)

        metric = load("accuracy")

        dataset = load_dataset(MATH.DATASET_PATH, name=MATH.DATASET_NAME)["test"]
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
            example_generator_kwargs={"mapping_fn": MATH.mapping_fn},
            generation_config=generation_config,
            **kwargs,
        )

        outputs = []
        for r in responses:
            predictions = Counter([_parse_label_from_solution(text) for text in r["responses"]])
            top_prediction = predictions.most_common(1)[0][0]
            label = r["label"]

            metric.add(predictions=bool(top_prediction == label), reference=1)
            outputs.append(
                {
                    "problem": r["input"]["text"],
                    "level": r["input"]["level"],
                    "type": r["input"]["type"],
                    "solution": r["label"],
                    "completion": top_prediction,
                    "passed": int(top_prediction == label),
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
