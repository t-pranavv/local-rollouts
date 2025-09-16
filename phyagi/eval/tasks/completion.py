# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from datasets import Dataset
from evaluate import load
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from phyagi.eval.distributed_utils import (
    all_gather_list,
    all_reduce_dict,
    is_main_process,
)
from phyagi.eval.generation import generate
from phyagi.utils.config import load_config
from phyagi.utils.file_utils import save_json_file

_ANSWER_LABEL_MAP = {"A": 1, "B": 2, "C": 3, "D": 4}


def _parse_completion_fn(text: str) -> str:
    match = re.search(r"(?:.*)(## Solution.*)", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_label_fn(completion: str) -> str:
    match = re.search(r"Answer:\s*([^\n\r]+)", completion, re.DOTALL)
    return match.group(1).strip() if match else ""


class Completion:
    """Completion evaluation task.

    This task evaluates the ability of a model to generate a completion given
    a prompt. Note that this task might not produce a metric if labels are not
    available.

    Format (JSONL):
        >>> {
        >>>   "id": ...,
        >>>   "source": ...,
        >>>   "question": ...,
        >>>   "answer": ...,
        >>>   "prompt": ...,
        >>> }

    Format (YAML):
        >>> - id: ...
        >>>   source: ...
        >>>   question: ...
        >>>   answer: ...
        >>>   prompt: ...
        >>>   ...

    """

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": example.get("id", 0),
            "problem_source": example.get("text", ""),
            "question": example.get("question", ""),
            "label": example.get("answer", ""),
            "text": example.get("prompt", ""),
        }

    @staticmethod
    def run(
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        input_file_path: Union[str, Path] = None,
        device: Optional[Union[int, torch.device]] = None,
        temperature: float = 0.01,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        stop_tokens: Optional[List[str]] = None,
        n_samples: int = 1,
        n_examples: Optional[int] = None,
        parse_completion_fn: Optional[Callable[[str], str]] = None,
        parse_label_fn: Optional[Callable[[str], str]] = None,
        answer_label_map: Optional[Dict[str, int]] = None,
        output_file_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if input_file_path is None:
            raise ValueError("`input_file_path` must be provided.")

        input_file_path = Path(input_file_path)
        if not input_file_path.exists():
            raise ValueError(f"`input_file_path` must be a valid path, but got '{input_file_path}'.")

        metric = load("accuracy")

        dataset = Dataset.from_list(load_config(input_file_path))
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        generation_config = {
            "n_samples": n_samples,
            "stop_tokens": stop_tokens or ["## Question"],
            "do_sample": True if temperature > 0 else False,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        }

        parse_completion_fn = parse_completion_fn or _parse_completion_fn
        parse_label_fn = parse_label_fn or _parse_label_fn
        answer_label_map = answer_label_map or _ANSWER_LABEL_MAP

        responses = generate(
            dataset,
            generation_engine="text_generation_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            example_generator_kwargs={"mapping_fn": Completion.mapping_fn},
            generation_config=generation_config,
            **kwargs,
        )

        outputs = []
        for r in responses:
            completions = [parse_completion_fn(text) for text in r["responses"]]
            completions_labels = [parse_label_fn(completion) for completion in completions]

            predictions = [answer_label_map.get(label, -1) for label in completions_labels]
            label = answer_label_map.get(r["label"], 0)

            metric.add_batch(predictions=predictions, references=[label] * n_samples)
            outputs.append(
                {
                    "id": r["input"]["id"],
                    "source": r["input"]["problem_source"],
                    "question": r["input"]["question"],
                    "answer": r["label"],
                    "prompt": r["input"]["text"],
                    "completion": completions,
                    "completion_label": completions_labels,
                    "correct": [p == label for p in predictions],
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
