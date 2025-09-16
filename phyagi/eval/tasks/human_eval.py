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


class HumanEval:
    """HumanEval evaluation task.

    Reference:
        Evaluating Large Language Models Trained on Code.
        https://arxiv.org/abs/2107.03374.

    """

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    DATASET_PATH = "openai_humaneval"
    DATASET_NAME = "openai_humaneval"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": example["task_id"],
            "text": example["prompt"],
            "label": example["test"] + "\n" + f"check({example['entry_point']})",
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
        pass_at_k: Optional[List[int]] = None,
        n_examples: Optional[int] = None,
        output_file_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Avoid error when multiprocessing from outside
        multiprocessing.set_start_method("fork", force=True)

        metric = load("code_eval")
        pass_at_k = pass_at_k or [1]

        dataset = load_dataset(HumanEval.DATASET_PATH, name=HumanEval.DATASET_NAME)["test"]
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
            example_generator_kwargs={"mapping_fn": HumanEval.mapping_fn},
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
