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


class _Arithmetic:
    """Arithmetic evaluation task.

    Reference:
        Language Models are Few-Shot Learners.
        https://arxiv.org/abs/2005.14165.

    """

    DATASET_PATH = "EleutherAI/arithmetic"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}{}"
        targets = [example["completion"]]

        return [
            {
                "text": prompt.format(example["context"], target),
                "target": target,
                "label": 1,
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
        metric = load("accuracy")

        dataset = load_dataset(_Arithmetic.DATASET_PATH, name=dataset_name)["validation"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        responses = generate(
            dataset,
            generation_engine="log_likelihood_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            example_generator_kwargs={"mapping_fn": _Arithmetic.mapping_fn},
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


class Arithmetic1DC(_Arithmetic):
    DATASET_NAME = "arithmetic_1dc"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _Arithmetic.run(Arithmetic1DC.DATASET_NAME, *args, **kwargs)


class Arithmetic2DA(_Arithmetic):
    DATASET_NAME = "arithmetic_2da"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _Arithmetic.run(Arithmetic2DA.DATASET_NAME, *args, **kwargs)


class Arithmetic2DM(_Arithmetic):
    DATASET_NAME = "arithmetic_2dm"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _Arithmetic.run(Arithmetic2DM.DATASET_NAME, *args, **kwargs)


class Arithmetic2DS(_Arithmetic):
    DATASET_NAME = "arithmetic_2ds"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _Arithmetic.run(Arithmetic2DS.DATASET_NAME, *args, **kwargs)


class Arithmetic3DA(_Arithmetic):
    DATASET_NAME = "arithmetic_3da"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _Arithmetic.run(Arithmetic3DA.DATASET_NAME, *args, **kwargs)


class Arithmetic3DS(_Arithmetic):
    DATASET_NAME = "arithmetic_3ds"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _Arithmetic.run(Arithmetic3DS.DATASET_NAME, *args, **kwargs)


class Arithmetic4DA(_Arithmetic):
    DATASET_NAME = "arithmetic_4da"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _Arithmetic.run(Arithmetic4DA.DATASET_NAME, *args, **kwargs)


class Arithmetic4DS(_Arithmetic):
    DATASET_NAME = "arithmetic_4ds"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _Arithmetic.run(Arithmetic4DS.DATASET_NAME, *args, **kwargs)


class Arithmetic5DA(_Arithmetic):
    DATASET_NAME = "arithmetic_5da"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _Arithmetic.run(Arithmetic5DA.DATASET_NAME, *args, **kwargs)


class Arithmetic5DS(_Arithmetic):
    DATASET_NAME = "arithmetic_5ds"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _Arithmetic.run(Arithmetic5DS.DATASET_NAME, *args, **kwargs)
