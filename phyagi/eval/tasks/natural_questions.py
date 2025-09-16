# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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


def _parse_targets_from_example(example: Dict[str, Any]) -> str:
    def _parse_target(target: Dict[str, Any], tokens: Dict[str, Any]) -> str:
        start_token = target["start_token"]
        end_token = target["end_token"]

        target_span = tokens["token"][start_token:end_token]
        target_is_html = tokens["is_html"][start_token:end_token]

        target_chars = [tok for (tok, is_html) in zip(target_span, target_is_html) if not is_html]
        target = " ".join(target_chars)

        return target

    targets = [
        {
            "start_token": start_token,
            "end_token": end_token,
        }
        for (start_token, end_token) in zip(
            example["long_answer_candidates"]["start_token"], example["long_answer_candidates"]["end_token"]
        )
    ]

    return [" " + _parse_target(target, example["document"]["tokens"]) for target in targets]


def _parse_label_from_example(example: Dict[str, Any]) -> str:
    for annotation in example["annotations"]["long_answer"]:
        candidate_index = annotation["candidate_index"]
        if candidate_index != -1:
            return candidate_index

    return -1


class NaturalQuestions:
    """Natural Questions evaluation task.

    Reference:
        Natural Questions: a Benchmark for Question Answering Research.
        https://research.google/pubs/pub47761.pdf.

    """

    DATASET_PATH = "natural_questions"
    DATASET_NAME = "dev"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = "{}{}"
        targets = _parse_targets_from_example(example)

        return [
            {
                "text": prompt.format(example["question"]["text"], target),
                "target": target,
                "label": _parse_label_from_example(example),
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
            "accuracy": load("accuracy"),
            "accuracy_norm": load("accuracy"),
        }

        dataset = load_dataset(NaturalQuestions.DATASET_PATH, name=NaturalQuestions.DATASET_NAME)["validation"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        responses = generate(
            dataset,
            generation_engine="log_likelihood_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            example_generator_kwargs={"mapping_fn": NaturalQuestions.mapping_fn},
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
