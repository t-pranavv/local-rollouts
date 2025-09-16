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


class MBPP:
    """MBPP evaluation task.

    Reference:
        Program Synthesis with Large Language Models.
        https://arxiv.org/abs/2108.07732.

    """

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    DATASET_PATH = "mbpp"
    DATASET_NAME = "sanitized"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example["prompt"]
        code_header = example["code"].split(":")[0] + ":"
        source_template = '{}\n\t"""\n\t{}\n\t{}\n\t"""\n\t'

        return {
            "task_id": example["task_id"],
            "prompt": prompt,
            "code_header": code_header,
            "text": source_template.format(code_header, prompt, "\n\t".join(example["test_list"])),
            "label": "\n".join(example["test_list"]),
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

        dataset = load_dataset(MBPP.DATASET_PATH, name=MBPP.DATASET_NAME)["test"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        generation_config = {
            "n_samples": n_samples,
            "stop_tokens": ["\nclass", "\ndef", "\nassert", "\n#", "\n@", "\nif", "\nprint"],
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
            example_generator_kwargs={"mapping_fn": MBPP.mapping_fn},
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
                    "prompt": r["input"]["prompt"],
                    "reference": r["label"],
                    "completion": predictions[0],
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
