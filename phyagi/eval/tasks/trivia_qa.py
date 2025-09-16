# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Optional, Union

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


class TriviaQA:
    """TriviaQA evaluation task.

    Reference:
        TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.
        https://arxiv.org/abs/1705.03551.

    """

    DATASET_PATH = "trivia_qa"
    DATASET_NAME = "rc"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}{}"
        targets = [" " + alias for alias in example["answer"]["aliases"]]

        return [
            {
                "text": prompt.format(example["question"], target),
                "target": target,
                "label": 1,
            }
            for target in targets
        ]

    @staticmethod
    def run(
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        n_examples: Optional[int] = None,
        output_file_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        metric = load("accuracy")

        dataset = load_dataset(TriviaQA.DATASET_PATH, name=TriviaQA.DATASET_NAME)["validation"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        responses = generate(
            dataset,
            generation_engine="log_likelihood_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            example_generator_kwargs={"mapping_fn": TriviaQA.mapping_fn},
            **kwargs,
        )

        outputs = []
        for r in responses:
            prediction = any(r["exact_matches"])
            label = r["labels"][0]

            metric.add(predictions=prediction, reference=label)
            outputs.append(r)

        results = metric.compute()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

            outputs = all_gather_list(outputs)
            results = all_reduce_dict(results)

        if is_main_process():
            save_json_file(outputs, output_file_path) if output_file_path else None

        return results
