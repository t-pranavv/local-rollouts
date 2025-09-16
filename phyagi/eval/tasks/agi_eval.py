# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import re
from collections import Counter
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


class _AGIEval:
    """AGIEval evaluation task.

    Reference:
        AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models.
        https://arxiv.org/pdf/2304.06364.pdf.

    """

    DATASET_PATH = "lighteval/agi_eval_en"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        def _parse_target(target: str) -> str:
            return re.sub(r"\([A-E]\)", "", target).strip()

        prompt = "{} {} {}"
        targets = [_parse_target(option) for option in example["options"]]

        return [
            {
                "text": prompt.format(example["passage"], example["question"], target),
                "target": target,
                "label": ["A", "B", "C", "D", "E"].index(example["label"]),
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
        **kwargs,
    ) -> Dict[str, Any]:
        metric = {
            "accuracy": load("accuracy"),
            "accuracy_norm": load("accuracy"),
        }

        dataset = load_dataset(_AGIEval.DATASET_PATH, name=dataset_name)["train"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        responses = generate(
            dataset,
            generation_engine="log_likelihood_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            example_generator_kwargs={"mapping_fn": _AGIEval.mapping_fn},
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


class AGIMATH(_AGIEval):
    DATASET_NAME = "math"

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    @staticmethod
    def _parse_answer(text: str) -> str:
        answer_regex = re.compile(r"Answer: (.*)")

        match = answer_regex.search(text)
        if match:
            return match.group(1).strip()

        return ""

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = "Solve the mathematical question with a numerical answer. Do not write extra information, only numbers or expressions.\nQuestion: {}\nAnswer:"
        return {
            "text": prompt.format(example["question"]),
            "label": example["answer"],
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

        dataset = load_dataset(AGIMATH.DATASET_PATH, name=AGIMATH.DATASET_NAME)["train"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        generation_config = {
            "n_samples": n_samples,
            "stop_tokens": ["\n", "\n\n"],
            "do_sample": True,
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
            example_generator_kwargs={"mapping_fn": AGIMATH.mapping_fn},
            generation_config=generation_config,
            **kwargs,
        )

        outputs = []
        for r in responses:
            preds = Counter([AGIMATH._parse_answer(text) for text in r["responses"]])
            top_pred = preds.most_common(1)[0][0]
            label = r["label"]

            metric.add(predictions=bool(top_pred == label), reference=1)
            outputs.append(
                {
                    "question": r["input"]["text"],
                    "answer": r["label"],
                    "completion": top_pred,
                    "passed": int(top_pred == label),
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


class AQuARAT(_AGIEval):
    DATASET_NAME = "aqua_rat"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _AGIEval.run(AQuARAT.DATASET_NAME, *args, **kwargs)


class LogiQAEn(_AGIEval):
    DATASET_NAME = "logiqa-en"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _AGIEval.run(LogiQAEn.DATASET_NAME, *args, **kwargs)


class LSATAR(_AGIEval):
    DATASET_NAME = "lsat-ar"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _AGIEval.run(LSATAR.DATASET_NAME, *args, **kwargs)


class LSATLR(_AGIEval):
    DATASET_NAME = "lsat-lr"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _AGIEval.run(LSATLR.DATASET_NAME, *args, **kwargs)


class LSATRC(_AGIEval):
    DATASET_NAME = "lsat-rc"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _AGIEval.run(LSATRC.DATASET_NAME, *args, **kwargs)


class SATEn(_AGIEval):
    DATASET_NAME = "sat-en"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _AGIEval.run(SATEn.DATASET_NAME, *args, **kwargs)


class SATMath(_AGIEval):
    DATASET_NAME = "sat-math"

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _AGIEval.run(SATMath.DATASET_NAME, *args, **kwargs)
