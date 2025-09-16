# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from phyagi.datasets.rl.chat.chat_dataset import ChatDataset


@pytest.fixture
def dummy_data():
    return {
        "convo": [
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}],
            [{"role": "user", "content": "What's the weather like today?"}],
            [{"role": "user", "content": "Tell me a very long story." * 100}],
        ],
        "gt": ["Hi there!", "It's sunny today.", "Once upon a time..."],
    }


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("microsoft/phi-4")


def test_chat_dataset(dummy_data, tokenizer):
    dataset = ChatDataset(
        dataset=Dataset.from_dict(dummy_data),
        tokenizer=tokenizer,
        messages_column_name="convo",
        ground_truth_column_name="gt",
        max_length=512,
        tokenize=True,
    )
    assert isinstance(dataset, ChatDataset)
    assert len(dataset) == 3

    sample = dataset[0]
    assert isinstance(sample, dict)
    assert set(sample.keys()) >= {"index", "messages", "raw_prompt", "prompt_input_ids", "ground_truth"}
    assert isinstance(sample["prompt_input_ids"], list)


def test_chat_dataset_filter_long_inputs(dummy_data, tokenizer):
    dataset = ChatDataset(
        dataset=Dataset.from_dict(dummy_data),
        tokenizer=tokenizer,
        messages_column_name="convo",
        ground_truth_column_name="gt",
        max_length=20,
        filter_max_length=True,
        tokenize=True,
    )
    assert len(dataset) < 3
    dataset.assert_within_max_length()


def test_chat_dataset_length_assert_failure(dummy_data, tokenizer):
    dataset = ChatDataset(
        dataset=Dataset.from_dict(dummy_data),
        tokenizer=tokenizer,
        messages_column_name="convo",
        ground_truth_column_name="gt",
        max_length=5,
        filter_max_length=False,
        tokenize=True,
    )
    with pytest.raises(ValueError):
        dataset.assert_within_max_length()


def test_chat_dataset_invalid_multiple_splits(tokenizer):
    dummy = DatasetDict(
        {
            "train": Dataset.from_dict({"messages": [["test"]], "gt": ["A"]}),
            "test": Dataset.from_dict({"messages": [["test"]], "gt": ["B"]}),
        }
    )

    with pytest.raises(ValueError):
        ChatDataset(
            dataset=dummy,
            tokenizer=tokenizer,
            messages_column_name="messages",
            ground_truth_column_name="gt",
            max_length=50,
        )
