# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Callable, Dict, List, Optional, Union

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


class _OpenBookQA:
    """OpenBookQA evaluation task.

    Reference:
        Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering.
        https://arxiv.org/pdf/1809.02789.pdf.

    """

    DATASET_PATH = "openbookqa"

    @staticmethod
    def run(
        dataset_name: str,
        mapping_fn: Callable,
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

        dataset = load_dataset(_OpenBookQA.DATASET_PATH, name=dataset_name)["test"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        responses = generate(
            dataset,
            generation_engine="log_likelihood_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            example_generator_kwargs={"mapping_fn": mapping_fn},
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


class OpenBookQAAdditional(_OpenBookQA):
    DATASET_NAME = "additional"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}{}"
        source = example["fact1"] + ". " + example["question_stem"]
        targets = [" " + choice for choice in example["choices"]["text"]]

        return [
            {
                "text": prompt.format(source, target),
                "target": target,
                "label": ["A", "B", "C", "D"].index(example["answerKey"].strip()),
            }
            for target in targets
        ]

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _OpenBookQA.run(OpenBookQAAdditional.DATASET_NAME, OpenBookQAAdditional.mapping_fn, *args, **kwargs)


class OpenBookQAMain(_OpenBookQA):
    DATASET_NAME = "main"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}{}"
        targets = [" " + choice for choice in example["choices"]["text"]]

        return [
            {
                "text": prompt.format(example["question_stem"], target),
                "target": target,
                "label": ["A", "B", "C", "D"].index(example["answerKey"].strip()),
            }
            for target in targets
        ]

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _OpenBookQA.run(OpenBookQAMain.DATASET_NAME, OpenBookQAMain.mapping_fn, *args, **kwargs)
