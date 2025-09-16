# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict, IterableDatasetDict, load_dataset
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from phyagi.eval.distributed_utils import all_gather_list
from phyagi.eval.generation import generate


class LossHFHubDataset:
    """Loss/perplexity (using datasets from Hugging Face hub) evaluation task."""

    @staticmethod
    def mapping_fn(example: Dict[str, Any], column_name: str, eos_token_id: int) -> List[Dict[str, Any]]:
        # Prevent an empty string from being passed to the model
        return {
            "text": example[column_name] if example[column_name] else str(eos_token_id),
        }

    @staticmethod
    def run(
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        dataset_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        split: Optional[str] = None,
        data_files: Optional[Dict[str, Union[str, List[str]]]] = None,
        column_name: str = "text",
        shift_labels: bool = True,
        n_examples: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        dataset = load_dataset(dataset_path, name=dataset_name, split=split, data_files=data_files)
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            raise ValueError(f"`dataset` must not be a dictionary of datasets, but got '{type(dataset)}'.")

        responses = generate(
            dataset,
            generation_engine="loss_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            example_generator_kwargs={
                "mapping_fn": LossHFHubDataset.mapping_fn,
                "column_name": column_name,
                "eos_token_id": tokenizer.eos_token_id,
            },
            shift_labels=shift_labels,
            **kwargs,
        )

        loss = []
        for r in responses:
            loss.append(r["loss"])

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            loss = all_gather_list(loss)

        return {"loss": np.mean(loss), "ppl": np.exp(np.mean(loss))}


class LossNumpyDataset:
    """Loss/perplexity (using datasets from NumPy arrays) evaluation task."""

    @staticmethod
    def mapping_fn(
        input_ids: List[int], seq_len: int
    ) -> Generator[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
        n_input_ids = ((len(input_ids) - 1) // seq_len) * seq_len + 1
        n_sequences = math.ceil((n_input_ids - 1) / seq_len)

        for i in range(n_sequences):
            idx = i * seq_len
            cur_len = min(seq_len, n_input_ids - 1 - idx)
            yield {"input_ids": input_ids[idx : (idx + cur_len)].astype(np.int64)}

    @staticmethod
    def run(
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[Union[int, torch.device]] = None,
        npy_file_path: Optional[str] = None,
        seq_len: int = 2048,
        shift_labels: bool = True,
        n_examples: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        dataset = Dataset.from_generator(
            LossNumpyDataset.mapping_fn,
            gen_kwargs={"input_ids": np.load(npy_file_path, mmap_mode="r"), "seq_len": seq_len},
        )
        dataset = dataset.select(range(n_examples)) if n_examples is not None else dataset

        responses = generate(
            dataset,
            generation_engine="loss_pipeline",
            model=model,
            tokenizer=tokenizer,
            device=device,
            shift_labels=shift_labels,
            **kwargs,
        )

        loss = []
        for r in responses:
            loss.append(r["loss"])

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            loss = all_gather_list(loss)

        return {"loss": np.mean(loss), "ppl": np.exp(np.mean(loss))}
