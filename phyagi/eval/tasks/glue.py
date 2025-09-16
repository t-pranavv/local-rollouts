# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
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


def _clean_text(text: str) -> str:
    text = text.replace(" n't", "n't")
    text = text.replace(" )", ")")
    text = text.replace("( ", "(")
    text = text.replace('" ', '"')
    text = text.replace(' "', '"')
    text = re.sub(r" (['.,])", r"\1", text)

    return text


class _GLUE:
    """GLUE evaluation task.

    Reference:
        GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding.
        https://openreview.net/pdf?id=rJ4km2R5t7.

    """

    DATASET_PATH = "glue"

    @staticmethod
    def run(
        dataset_name: str,
        dataset_split: str,
        mapping_fn: Callable,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        n_examples: Optional[int] = None,
        output_file_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        metric = load(_GLUE.DATASET_PATH, config_name=dataset_name)

        dataset = load_dataset(_GLUE.DATASET_PATH, name=dataset_name, split=dataset_split)
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
            label = r["labels"][0]

            prediction = np.argmax(log_likelihoods)
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


class CoLA(_GLUE):
    DATASET_NAME = "cola"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}\nQuestion: Does this sentence make sense?\nAnswer:{}"
        targets = [" no", " yes"]

        return [
            {
                "text": prompt.format(example["sentence"], target),
                "target": target,
                "label": example["label"],
            }
            for target in targets
        ]

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _GLUE.run(
            CoLA.DATASET_NAME,
            "validation",
            CoLA.mapping_fn,
            *args,
            **kwargs,
        )


class MNLI(_GLUE):
    DATASET_NAME = "mnli"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}\nQuestion: {} True, False or Neither?\nAnswer:{}"

        hypothesis = example["hypothesis"].strip() + ("" if example["hypothesis"].strip().endswith(".") else ".")
        targets = [" True", " Neither", " False"]

        return [
            {
                "text": prompt.format(example["premise"], hypothesis, target),
                "target": target,
                "label": example["label"],
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
        metric = {
            "matched_accuracy": load(MNLI.DATASET_PATH, config_name=MNLI.DATASET_NAME),
            "mismatched_accuracy": load(MNLI.DATASET_PATH, config_name=MNLI.DATASET_NAME),
        }

        dataset = {
            "matched": load_dataset(_GLUE.DATASET_PATH, name=MNLI.DATASET_NAME, split="validation_matched"),
            "mismatched": load_dataset(_GLUE.DATASET_PATH, name=MNLI.DATASET_NAME, split="validation_mismatched"),
        }
        dataset = {
            key: dataset[key].select(range(n_examples)) if n_examples is not None else dataset[key] for key in dataset
        }

        outputs = {
            "matched": [],
            "mismatched": [],
        }
        for d, m, o in zip(dataset.values(), metric.values(), outputs.values()):
            responses = generate(
                d,
                generation_engine="log_likelihood_pipeline",
                model=model,
                tokenizer=tokenizer,
                device=device,
                example_generator_kwargs={"mapping_fn": MNLI.mapping_fn},
                **kwargs,
            )

            for r in responses:
                log_likelihoods = r["log_likelihoods"]
                label = r["labels"][0]

                prediction = np.argmax(log_likelihoods)
                m.add(predictions=prediction, reference=label)

                o.append(r)

        results = {key: metric.compute()["accuracy"] for key, metric in metric.items()}

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

            for key in outputs.keys():
                outputs[key] = all_gather_list(outputs[key])
            results = all_reduce_dict(results)

        if is_main_process():
            save_json_file(outputs, output_file_path) if output_file_path else None

        return results


class MRPC(_GLUE):
    DATASET_NAME = "mrpc"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "Sentence 1: {}\nSentence 2: {}\nQuestion: Do both sentences mean the same thing?\nAnswer:{}"

        sentence_1 = _clean_text(example["sentence1"])
        sentence_2 = _clean_text(example["sentence2"])
        targets = [" no", " yes"]

        return [
            {
                "text": prompt.format(sentence_1, sentence_2, target),
                "target": target,
                "label": example["label"],
            }
            for target in targets
        ]

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _GLUE.run(
            MRPC.DATASET_NAME,
            "validation",
            MRPC.mapping_fn,
            *args,
            **kwargs,
        )


class QNLI(_GLUE):
    DATASET_NAME = "qnli"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}\n{}\nQuestion: Does this response answer the question?\nAnswer:{}"
        targets = [" yes", " no"]

        return [
            {
                "text": prompt.format(example["question"], example["sentence"], target),
                "target": target,
                "label": example["label"],
            }
            for target in targets
        ]

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _GLUE.run(
            QNLI.DATASET_NAME,
            "validation",
            QNLI.mapping_fn,
            *args,
            **kwargs,
        )


class QQP(_GLUE):
    DATASET_NAME = "qqp"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "Question 1: {}\nQuestion 2: {}\nQuestion: Do both questions ask the same thing?\nAnswer:{}"
        targets = [" no", " yes"]

        return [
            {
                "text": prompt.format(example["question1"], example["question2"], target),
                "target": target,
                "label": example["label"],
            }
            for target in targets
        ]

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _GLUE.run(
            QQP.DATASET_NAME,
            "validation",
            QQP.mapping_fn,
            *args,
            **kwargs,
        )


class RTE(_GLUE):
    DATASET_NAME = "rte"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}\nQuestion: {} True or False?\nAnswer:"
        targets = [" True", " False"]

        return [
            {
                "text": prompt.format(example["sentence1"], example["sentence2"], target),
                "target": target,
                "label": example["label"],
            }
            for target in targets
        ]

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _GLUE.run(
            RTE.DATASET_NAME,
            "validation",
            RTE.mapping_fn,
            *args,
            **kwargs,
        )


class SST2(_GLUE):
    DATASET_NAME = "sst2"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}\nQuestion: Is this sentence positive or negative?\nAnswer:{}"

        sentence = _clean_text(example["sentence"])
        targets = [" negative", " positive"]

        return [
            {
                "text": prompt.format(sentence, target),
                "target": target,
                "label": example["label"],
            }
            for target in targets
        ]

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _GLUE.run(
            SST2.DATASET_NAME,
            "validation",
            SST2.mapping_fn,
            *args,
            **kwargs,
        )


class WNLI(_GLUE):
    DATASET_NAME = "wnli"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}\nQuestion: {} True or False?\nAnswer:{}"
        targets = [" False", " True"]

        return [
            {
                "text": prompt.format(example["sentence1"], example["sentence2"], target),
                "target": target,
                "label": example["label"],
            }
            for target in targets
        ]

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _GLUE.run(
            WNLI.DATASET_NAME,
            "validation",
            WNLI.mapping_fn,
            *args,
            **kwargs,
        )
