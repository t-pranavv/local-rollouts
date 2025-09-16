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


class _LAMBADA:
    """LAMBADA evaluation task.

    Reference:
        The LAMBADA dataset: Word prediction requiring a broad discourse context.
        https://arxiv.org/pdf/1606.06031.pdf.

    """

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}{}"
        context, target = example["text"].rsplit(" ", 1)

        return [
            {
                "text": prompt.format(context, target),
                "target": " " + target,
                "label": 1,
            }
        ]

    @staticmethod
    def run(
        dataset_name: str,
        dataset_path: str,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        n_examples: Optional[int] = None,
        output_file_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        metric = load("accuracy")

        dataset = load_dataset(dataset_name, name=dataset_path)["test"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        responses = generate(
            dataset,
            generation_engine="log_likelihood_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            example_generator_kwargs={"mapping_fn": _LAMBADA.mapping_fn},
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


class LAMBADA(_LAMBADA):
    DATASET_PATH = "lambada"
    DATASET_NAME = "plain_text"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _LAMBADA.run(LAMBADA.DATASET_PATH, LAMBADA.DATASET_NAME, *args, **kwargs)


class LAMBADAOpenAI(_LAMBADA):
    DATASET_PATH = "craffel/openai_lambada"
    DATASET_NAME = None

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _LAMBADA.run(LAMBADAOpenAI.DATASET_PATH, LAMBADAOpenAI.DATASET_NAME, *args, **kwargs)
