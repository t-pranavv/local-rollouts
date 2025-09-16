# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Optional, Union

import numpy as np
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


class _ARC:
    """ARC evaluation task.

    Reference:
        Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.
        https://arxiv.org/pdf/1803.05457.pdf.

    """

    DATASET_PATH = "ai2_arc"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "Question: {}\nAnswer:{}"
        targets = [" " + choice for choice in example["choices"]["text"]]

        # Prevents `label` from having wrong values due to dataset values
        # mixed between strings and integers
        answer_key_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        answer_key = answer_key_map.get(example["answerKey"], example["answerKey"])

        return [
            {
                "text": prompt.format(example["question"], target),
                "target": target,
                "label": ["A", "B", "C", "D", "E"].index(answer_key),
            }
            for target in targets
        ]

    @staticmethod
    def run(
        dataset_name: str,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        n_examples: Optional[int] = None,
        output_file_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        metric = {
            "accuracy": load("accuracy"),
            "accuracy_norm": load("accuracy"),
        }

        dataset = load_dataset(_ARC.DATASET_PATH, name=dataset_name)["test"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        responses = generate(
            dataset,
            generation_engine="log_likelihood_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            example_generator_kwargs={"mapping_fn": _ARC.mapping_fn},
            **kwargs,
        )

        outputs = []
        for r in responses:
            log_likelihoods = r["log_likelihoods"]
            target_lengths = r["target_lengths"]
            label = r["labels"][0]

            prediction = np.argmax(log_likelihoods)
            prediction_norm = np.argmax(np.array(log_likelihoods) / target_lengths)

            metric["accuracy"].add(predictions=prediction, reference=label)
            metric["accuracy_norm"].add(predictions=prediction_norm, reference=label)

            outputs.append(r)

        results = {key: metric.compute()["accuracy"] for key, metric in metric.items()}

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

            outputs = all_gather_list(outputs)
            results = all_reduce_dict(results)

        if is_main_process():
            save_json_file(outputs, output_file_path) if output_file_path else None

        return results


class ARCChallenge(_ARC):
    DATASET_NAME = "ARC-Challenge"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _ARC.run(ARCChallenge.DATASET_NAME, *args, **kwargs)


class ARCEasy(_ARC):
    DATASET_NAME = "ARC-Easy"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _ARC.run(ARCEasy.DATASET_NAME, *args, **kwargs)
