# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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


class SQuAD:
    """SQuAD evaluation task.

    Reference:
        Know What You Don't Know: Unanswerable Questions for SQuAD.
        https://arxiv.org/abs/1806.03822.

    """

    DATASET_PATH = "squad"
    DATASET_NAME = None

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": example["id"],
            "text": f"Title: {example['title']}\n\nContext: {example['context']}\n\nQuestion: {example['question']}\n\n Answer:",
            "label": example["answers"]["text"],
        }

    @staticmethod
    def run(
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        temperature: float = 0.01,
        top_p: float = 0.95,
        max_new_tokens: int = 300,
        n_samples: int = 1,
        n_examples: Optional[int] = None,
        output_file_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        metric = load("accuracy")

        dataset = load_dataset(SQuAD.DATASET_PATH, name=SQuAD.DATASET_NAME)["validation"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        generation_config = {
            "n_samples": n_samples,
            "stop_tokens": ["\n"],
            "do_sample": True if temperature > 0 else False,
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
            example_generator_kwargs={"mapping_fn": SQuAD.mapping_fn},
            generation_config=generation_config,
            **kwargs,
        )

        outputs = []
        for r in responses:
            predictions = [text.split(r["input"]["text"])[-1] for text in r["responses"]]
            label = r["label"]

            matches = any([any(s in p for s in label) for p in predictions])
            metric.add(prediction=matches, reference=1)

            outputs.append(
                {
                    "id": r["input"]["id"],
                    "prompt": r["input"]["text"],
                    "reference": r["label"],
                    "prediction": predictions,
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
