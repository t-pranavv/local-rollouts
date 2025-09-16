# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import multiprocessing
import os
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


class _HumanEvalPack:
    """HumanEval (pack) evaluation task.

    Reference:
        OctoPack: Instruction Tuning Code Large Language Models.
        https://arxiv.org/abs/2308.07124

    """

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    DATASET_PATH = "bigcode/humanevalpack"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": example["task_id"],
            "text": example["prompt"],
            "label": example["test"],
        }

    @staticmethod
    def run(
        dataset_name: str,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        temperature: Optional[float] = 0.01,
        top_p: Optional[float] = 0.95,
        max_new_tokens: Optional[int] = 300,
        n_samples: Optional[int] = 1,
        pass_at_k: Optional[List[int]] = None,
        n_examples: Optional[int] = None,
        output_file_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Avoid error when multiprocessing from outside
        multiprocessing.set_start_method("fork", force=True)

        metric = load("Muennighoff/code_eval_octopack")
        pass_at_k = pass_at_k or [1]

        dataset = load_dataset(_HumanEvalPack.DATASET_PATH, dataset_name)["test"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        generation_config = {
            "n_samples": n_samples,
            "stop_tokens": ["\nclass", "\ndef", "\n#", "\n@", "\nif", "\nprint"],
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
            example_generator_kwargs={"mapping_fn": _HumanEvalPack.mapping_fn},
            generation_config=generation_config,
            **kwargs,
        )

        outputs = []
        for r in responses:
            predictions = [list(r["responses"])]
            label = [r["label"]]

            metric.add_batch(predictions=predictions, references=label)
            outputs.append(
                {
                    "task_id": r["input"]["task_id"],
                    "prompt": r["input"]["text"],
                    "reference": r["label"],
                    "completion": [text.split(r["input"]["text"])[-1] for text in r["responses"]],
                }
            )

        results, tasks = metric.compute(k=pass_at_k, num_workers=1)
        for task_id, task_data in tasks.items():
            outputs[task_id].update({"passed": [int(completion[1]["passed"]) for completion in task_data]})

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

            outputs = all_gather_list(outputs)
            results = all_reduce_dict(results)

        if is_main_process():
            save_json_file(outputs, output_file_path) if output_file_path else None

        return results


class HumanEvalPackPython(_HumanEvalPack):
    DATASET_NAME = "python"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _HumanEvalPack.run(HumanEvalPackPython.DATASET_NAME, *args, **kwargs)


class HumanEvalPackJS(_HumanEvalPack):
    DATASET_NAME = "javascript"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _HumanEvalPack.run(HumanEvalPackJS.DATASET_NAME, *args, **kwargs)
