# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tempfile
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from phyagi.datasets.shared_memory_utils import (
    _SHMArray,
    process_dataset_to_memory,
    save_memory_dataset,
)


@pytest.fixture
def dataset_dict():
    return DatasetDict(
        {
            "train": Dataset.from_dict({"input": ["hello world", "my name is John"], "length": [[2], [4]]}),
            "test": Dataset.from_dict({"input": ["goodbye world", "see you later"], "length": [[2], [3]]}),
        }
    )


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


@pytest.fixture
def cache_dir():
    with tempfile.TemporaryDirectory() as cache_dir:
        yield Path(cache_dir)


def test_process_dataset_to_memory(dataset_dict, cache_dir):
    dtype = np.int64
    processing_column_name = ["length"]

    processed_dataset_dict = process_dataset_to_memory(dataset_dict, cache_dir, dtype, processing_column_name)

    assert isinstance(processed_dataset_dict, dict)
    for key in processed_dataset_dict.keys():
        assert isinstance(processed_dataset_dict[key], _SHMArray)
        assert processed_dataset_dict[key].dtype == dtype


def test_save_memory_dataset(dataset_dict, tokenizer, cache_dir):
    dtype = np.int64
    processing_column_name = ["length"]

    processed_dataset_dict = process_dataset_to_memory(dataset_dict, cache_dir, dtype, processing_column_name)

    cache_files = save_memory_dataset(processed_dataset_dict, tokenizer, cache_dir)

    assert isinstance(cache_files, dict)
    for key in cache_files.keys():
        if "tokenizer" in key:
            assert cache_files[key].suffix == ".pkl"
        else:
            assert cache_files[key].suffix == ".npy"
