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


class _SuperGLUE:
    """SuperGLUE evaluation task.

    Reference:
        SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems.
        https://w4ngatang.github.io/static/papers/superglue.pdf.

    """

    DATASET_PATH = "super_glue"

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
        metric = load(_SuperGLUE.DATASET_PATH, config_name=dataset_name)

        dataset = load_dataset(_SuperGLUE.DATASET_PATH, name=dataset_name, split=dataset_split)
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


class AXb(_SuperGLUE):
    DATASET_NAME = "axb"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}\nQuestion: {} True or False?\nAnswer:{}"
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
        return _SuperGLUE.run(
            AXb.DATASET_NAME,
            "test",
            AXb.mapping_fn,
            *args,
            **kwargs,
        )


class AXg(_SuperGLUE):
    DATASET_NAME = "axg"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}\nQuestion: {} True or False?\nAnswer:{}"
        targets = [" True", " False"]

        return [
            {
                "text": prompt.format(example["premise"], example["hypothesis"], target),
                "target": target,
                "label": example["label"],
            }
            for target in targets
        ]

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _SuperGLUE.run(
            AXg.DATASET_NAME,
            "test",
            AXg.mapping_fn,
            *args,
            **kwargs,
        )


class BoolQ(_SuperGLUE):
    DATASET_NAME = "boolq"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}\nQuestion: {}?\nAnswer:{}"
        targets = [" no", " yes"]

        return [
            {
                "text": prompt.format(example["passage"], example["question"], target),
                "target": target,
                "label": example["label"],
            }
            for target in targets
        ]

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _SuperGLUE.run(
            BoolQ.DATASET_NAME,
            "validation",
            BoolQ.mapping_fn,
            *args,
            **kwargs,
        )


class CB(_SuperGLUE):
    DATASET_NAME = "cb"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}\nQuestion: {}. True, False or Neither?\nAnswer:{}"
        targets = [" True", " False", " Neither"]

        return [
            {
                "text": prompt.format(example["premise"], example["hypothesis"], target),
                "target": target,
                "label": example["label"],
            }
            for target in targets
        ]

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _SuperGLUE.run(
            CB.DATASET_NAME,
            "validation",
            CB.mapping_fn,
            *args,
            **kwargs,
        )


class COPA(_SuperGLUE):
    DATASET_NAME = "copa"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        def _get_target_from_choice(choice: str) -> str:
            return choice[0].lower() + choice[1:]

        prompt = "{} {}{}"
        connector = {"cause": "because", "effect": "therefore"}

        question = connector[example["question"]]
        premise = example["premise"].strip()[:-1]
        targets = [" " + _get_target_from_choice(example["choice1"]), " " + _get_target_from_choice(example["choice2"])]

        return [
            {
                "text": prompt.format(question, premise, target),
                "target": target,
                "label": example["label"],
            }
            for target in targets
        ]

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _SuperGLUE.run(
            COPA.DATASET_NAME,
            "validation",
            COPA.mapping_fn,
            *args,
            **kwargs,
        )


class MultiRC(_SuperGLUE):
    DATASET_NAME = "multirc"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        def _get_target_from_answer(answer: str, label: bool) -> str:
            label = "yes" if label else "no"
            return f" {answer}\nIs the answer correct? {label}"

        prompt = "{}\nQuestion: {}\nAnswer:{}"
        targets = [_get_target_from_answer(example["answer"], False), _get_target_from_answer(example["answer"], True)]

        return [
            {
                "idx": example["idx"],
                "text": prompt.format(example["paragraph"], example["question"], target),
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
        metric = load(MultiRC.DATASET_PATH, config_name=MultiRC.DATASET_NAME)

        dataset = load_dataset(MultiRC.DATASET_PATH, name=MultiRC.DATASET_NAME)["validation"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        responses = generate(
            dataset,
            generation_engine="log_likelihood_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            example_generator_kwargs={"mapping_fn": MultiRC.mapping_fn},
            **kwargs,
        )

        outputs = []
        for r in responses:
            prediction = {"idx": r["inputs"][0]["idx"], "prediction": np.argmax(r["log_likelihoods"])}
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


class ReCoRD(_SuperGLUE):
    DATASET_NAME = "record"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        def _get_input_from_example(example: Dict[str, Any]) -> str:
            passage, *highlights = example["passage"].strip().split("\n@highlight\n")

            text = passage + "\n\n"
            for highlight in highlights:
                text += f"  - {highlight}.\n"

            return text

        def _get_target_from_query(query: str, entity: str) -> str:
            return f"  - {query}".replace("@placeholder", entity)

        prompt = "{}{}"
        targets = sorted(list(set(example["entities"])))
        answers = sorted(list(set(example["answers"])))

        return [
            {
                "idx": example["idx"],
                "entities": targets,
                "answers": answers,
                "text": prompt.format(
                    _get_input_from_example(example), _get_target_from_query(example["query"], target)
                ),
                "target": _get_target_from_query(example["query"], target),
                "label": None,
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
        metric = load(ReCoRD.DATASET_PATH, config_name=ReCoRD.DATASET_NAME)

        dataset = load_dataset(ReCoRD.DATASET_PATH, name=ReCoRD.DATASET_NAME)["validation"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        responses = generate(
            dataset,
            generation_engine="log_likelihood_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            example_generator_kwargs={"mapping_fn": ReCoRD.mapping_fn},
            **kwargs,
        )

        outputs = []
        for r in responses:
            prediction_idx = np.argmax(np.array(r["log_likelihoods"]))
            prediction = {
                "idx": r["inputs"][0]["idx"],
                "prediction_text": r["inputs"][0]["entities"][prediction_idx],
            }
            label = {"idx": r["inputs"][0]["idx"], "answers": r["inputs"][0]["answers"]}

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


class WiC(_SuperGLUE):
    DATASET_NAME = "wic"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "Sentence 1: {}\nSentence 2: {}\nQuestion: Is the word '{}' used in the same way in the two sentences above?\nAnswer:{}"
        word = example["sentence1"][example["start1"] : example["end1"]]
        targets = [" no", " yes"]

        return [
            {
                "text": prompt.format(example["sentence1"], example["sentence2"], word, target),
                "target": target,
                "label": example["label"],
            }
            for target in targets
        ]

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _SuperGLUE.run(
            WiC.DATASET_NAME,
            "validation",
            WiC.mapping_fn,
            *args,
            **kwargs,
        )


class WSC(_SuperGLUE):
    DATASET_NAME = "wsc"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        def _clean_text(text: str) -> str:
            text = text.replace(" n't", "n't")
            text = text.replace(" )", ")")
            text = text.replace("( ", "(")
            text = text.replace('" ', '"')
            text = text.replace(' "', '"')
            text = re.sub(r" (['.,])", r"\1", text)

            return text

        prompt = "Passage: {}\nQuestion: In the passage above, does the pronoun '{}' refer to '{}'?\nAnswer:{}"

        pre_passage = " ".join(example["text"].split()[: example["span2_index"]])
        post_passage = example["text"][len(pre_passage) + len(example["span2_text"]) + 1 :]
        passage = _clean_text(f"{pre_passage} *{example['span2_text']}*{post_passage}")

        pronoun = example["span2_text"]
        noun = example["span1_text"]
        targets = [" no", " yes"]

        return [
            {
                "text": prompt.format(passage, pronoun, noun, target),
                "target": target,
                "label": example["label"],
            }
            for target in targets
        ]

    @staticmethod
    def run(*args, **kwargs) -> Dict[str, Any]:
        return _SuperGLUE.run(
            WSC.DATASET_NAME,
            "validation",
            WSC.mapping_fn,
            *args,
            **kwargs,
        )
