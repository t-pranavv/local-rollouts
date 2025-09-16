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


class _RACE:
    """RACE evaluation task.

    Reference:
        RACE: Large-scale ReAding Comprehension Dataset From Examinations.
        https://arxiv.org/abs/1704.04683.

    """

    DATASET_PATH = "race"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{} {}{}"
        targets = [" " + option for option in example["options"]]

        return [
            {
                "text": prompt.format(example["article"], example["question"], target),
                "target": target,
                "label": ["A", "B", "C", "D"].index(example["answer"]),
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

        dataset = load_dataset(_RACE.DATASET_PATH, name=dataset_name)["test"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        responses = generate(
            dataset,
            generation_engine="log_likelihood_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            example_generator_kwargs={"mapping_fn": _RACE.mapping_fn},
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


class RACEHigh(_RACE):
    DATASET_NAME = "high"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _RACE.run(RACEHigh.DATASET_NAME, *args, **kwargs)


class RACEMiddle(_RACE):
    DATASET_NAME = "middle"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _RACE.run(RACEMiddle.DATASET_NAME, *args, **kwargs)
