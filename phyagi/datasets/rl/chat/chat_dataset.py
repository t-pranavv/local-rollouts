# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, Optional, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import DatasetDict as HfDatasetDict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class ChatDataset(Dataset):
    """Chat dataset."""

    def __init__(
        self,
        dataset: Union[HfDataset, HfDatasetDict],
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        messages_column_name: str = "messages",
        ground_truth_column_name: Optional[str] = None,
        max_length: Optional[int] = None,
        filter_max_length: bool = False,
        tokenize: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            dataset: Dataset or dataset dictionary.
            tokenizer: Tokenizer to use for encoding the dataset.
            messages_column_name: Column name for messages in the dataset.
            ground_truth_column_name: Column name for ground truth in the dataset.
            max_length: Maximum length of the input sequences.
            filter_max_length: Whether to filter sequences longer than ``max_length``.
            tokenize: Whether to pre-tokenize the dataset.

        """

        self._tokenizer = tokenizer
        self._max_length = max_length
        self._filter_max_length = max_length if filter_max_length else None
        self._tokenize = tokenize
        self._removed_rows = 0

        if isinstance(dataset, HfDatasetDict):
            if len(dataset.keys()) != 1:
                raise ValueError("`dataset` must be either a dataset or a dataset dictionary with a single split.")
            dataset = list(dataset.values())[0]

        rename_map = {messages_column_name: "messages"}
        if ground_truth_column_name:
            rename_map[ground_truth_column_name] = "ground_truth"
        dataset = dataset.rename_columns(rename_map)

        if self._tokenize:
            dataset = self._tokenize_dataset(dataset)

        self._data = dataset

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._data[idx]
        messages = item["messages"]

        input_ids = (
            self._tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if not self._tokenize
            else item["prompt_input_ids"]
        )

        if self._max_length is not None:
            if len(input_ids) > self._max_length:
                raise ValueError(
                    f"Length of `input_ids` should be smaller or equal to `max_length`, but got {len(input_ids)} and {self._max_length}. Use `filter_max_length=True` to filter the dataset."
                )

        return {
            "index": idx,
            "messages": messages.tolist() if isinstance(messages, np.ndarray) else messages,
            "raw_prompt": item.get("raw_prompt", ""),
            "prompt_input_ids": input_ids,
            "ground_truth": item.get("ground_truth"),
        }

    def _tokenize_dataset(self, data: HfDataset, num_proc: int = 1) -> HfDataset:
        def __tokenize_dataset(row: Dict[str, Any]) -> Dict[str, Any]:
            row["raw_prompt"] = self._tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True
            )
            row["prompt_input_ids"] = self._tokenizer(row["raw_prompt"], add_special_tokens=False)["input_ids"]
            row["prompt_token_counts"] = len(row["prompt_input_ids"])

            return row

        data = data.map(__tokenize_dataset, num_proc=num_proc, desc="Tokenizing dataset...")

        if self._filter_max_length is not None:
            original_nrows = len(data)

            data = data.filter(
                lambda x: x["prompt_token_counts"] <= self._filter_max_length,
                desc=f"Filtering prompts longer than {self._filter_max_length} tokens...",
            )
            if len(data) == 0:
                raise ValueError(
                    f"All prompts are longer than {self._filter_max_length} tokens. Set `filter_max_length=False` or increase `max_length`."
                )

            self._removed_rows = original_nrows - len(data)

        return data

    def assert_within_max_length(self) -> None:
        if self._max_length is None:
            return

        if not self._tokenize:
            raise ValueError("`tokenize` must be True when using `assert_within_max_length()`, but got False.")

        max_tokens = max(self._data["prompt_token_counts"])
        if max_tokens > self._max_length:
            raise ValueError(
                f"Found prompts with up to {max_tokens} tokens, but `max_length` is {self._max_length}. Use `filter_max_length=True` to filter the dataset or increase `max_length`."
            )
