# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Optional, Tuple, Union

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


def _parse_sentence_from_option(sentence: str, option: str) -> str:
    pronoun_loc = sentence.index("_")
    return sentence[:pronoun_loc] + option


def _parse_inputs_from_example(example: Dict[str, Any]) -> Tuple[str, str]:
    option = example["option" + example["answer"]]
    context = _parse_sentence_from_option(example["sentence"], option)

    pronoun_loc = example["sentence"].index("_") + 1
    target = " " + example["sentence"][pronoun_loc:].strip()

    return context, target


class WinoGrande:
    """WinoGrande evaluation task.

    Reference:
        WinoGrande: An Adversarial Winograd Schema Challenge at Scale.
        https://arxiv.org/pdf/1907.10641.pdf.

    """

    DATASET_PATH = "winogrande"
    DATASET_NAME = "winogrande_xl"

    @staticmethod
    def mapping_fn(example: Dict[str, Any]) -> List[Dict[str, Any]]:
        context, target = _parse_inputs_from_example(example)

        sources, options = [], [example["option1"], example["option2"]]
        for option in options:
            sent_inputs = _parse_sentence_from_option(example["sentence"], option)

            option_inputs = context.split("\n\n")
            option_inputs.pop()
            option_inputs = "\n\n".join([*option_inputs, sent_inputs]) if option_inputs else sent_inputs

            sources.append(option_inputs)

        return [
            {
                "text": source + target,
                "target": target,
                "label": 0 if example["answer"] == "1" else 1,
            }
            for source in sources
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
        metric = load("accuracy")

        dataset = load_dataset(WinoGrande.DATASET_PATH, name=WinoGrande.DATASET_NAME)["validation"]
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        responses = generate(
            dataset,
            generation_engine="log_likelihood_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            example_generator_kwargs={"mapping_fn": WinoGrande.mapping_fn},
            **kwargs,
        )

        outputs = []
        for r in responses:
            prediction = np.argmax(r["log_likelihoods"])
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
