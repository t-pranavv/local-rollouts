# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import pytest
import torch
from torch.utils.data import IterableDataset

from phyagi.datasets.concat_dataset import (
    SequentialWeightedConcatDataset,
    WeightedConcatChatDataset,
    WeightedConcatDataset,
    WeightedConcatIterableDataset,
)
from phyagi.datasets.rl.chat.chat_dataset import ChatDataset
from phyagi.datasets.train.lm.lm_dataset import LMDataset


class MyIterableDataset(IterableDataset):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class MyChatDataset(ChatDataset):
    def __init__(self, data, label=None):
        self._data = data
        self.label = label
        self.seq_len = 1

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return {"input_ids": self._data[idx]}


def test_weighted_concat_dataset():
    input_ids1 = np.array([1, 2, 3, 4, 5])
    input_ids2 = np.array([6, 7, 8, 9, 10])
    dataset1 = LMDataset(input_ids1, seq_len=1)
    dataset2 = LMDataset(input_ids2, seq_len=1)

    concat_dataset = WeightedConcatDataset([dataset1, dataset2], [1.0, 1.0])
    assert len(concat_dataset) == 8
    assert concat_dataset.cumulative_sizes == [4, 8]

    concat_dataset2 = WeightedConcatDataset([dataset1, dataset2], [0.25, 0.75])
    assert len(concat_dataset2) == 4
    assert concat_dataset2.cumulative_sizes == [1, 4]


def test_weighted_concat_dataset_invalid_input_lengths():
    dataset1 = LMDataset(np.array([1, 2]), seq_len=1)
    with pytest.raises(ValueError):
        WeightedConcatDataset([dataset1], [0.5, 0.5])


def test_weighted_concat_dataset_zero_weight():
    dataset1 = LMDataset(np.array([1, 2, 3]), seq_len=1)
    dataset2 = LMDataset(np.array([4, 5, 6]), seq_len=1)
    concat_dataset = WeightedConcatDataset([dataset1, dataset2], [1.0, 0.0])
    input_ids = [sample["input_ids"].item() for sample in concat_dataset]
    assert all(i in [1, 2, 3] for i in input_ids)


def test_weighted_concat_dataset_empty_datasets():
    with pytest.raises(ValueError):
        WeightedConcatDataset([], [])


def test_sequential_weighted_concat_dataset():
    input_ids1 = np.array([1, 2, 3, 4, 5])
    input_ids2 = np.array([6, 7, 8, 9, 10])
    dataset1 = LMDataset(input_ids1, seq_len=1)
    dataset2 = LMDataset(input_ids2, seq_len=1)

    concat_dataset = SequentialWeightedConcatDataset([dataset1, dataset2], [1.0, 1.0])
    assert len(concat_dataset) == 8
    assert concat_dataset.cumulative_sizes == [4, 8]
    assert concat_dataset[0]["input_ids"].item() == 1

    concat_dataset2 = SequentialWeightedConcatDataset([dataset1, dataset2], [0.5, 0.75])
    assert len(concat_dataset2) == 5
    assert concat_dataset2.cumulative_sizes == [2, 5]
    assert concat_dataset2[0]["input_ids"].item() == 1
    assert concat_dataset2[1]["input_ids"].item() == 2
    assert concat_dataset2[2]["input_ids"].item() == 6


def test_weighted_concat_iterable_dataset():
    dataset1 = MyIterableDataset([torch.tensor([i]) for i in range(10)])
    dataset2 = MyIterableDataset([torch.tensor([i]) for i in range(20, 30)])
    dataset3 = MyIterableDataset([torch.tensor([i]) for i in range(40, 50)])
    weights = [0.2, 0.3, 0.5]

    dataset = WeightedConcatIterableDataset([dataset1, dataset2, dataset3], weights)
    dataset.set_rank_and_world_size(0, 1)
    dataset_iter = iter(dataset)

    assert len(dataset._datasets) == 3
    assert np.array_equal(dataset._weights, np.array(weights))

    samples = []
    while True:
        try:
            samples.append(next(dataset_iter))
        except RuntimeError:
            break
    assert all(isinstance(sample, torch.Tensor) for sample in samples)


def test_weighted_concat_rl_dataset():
    dataset1 = MyChatDataset(np.array([1, 2, 3, 4]))
    dataset2 = MyChatDataset(np.array([10, 20, 30]))

    concat_dataset = WeightedConcatChatDataset(
        datasets=[dataset1, dataset2],
        weights=[1.0, 1.0],
        labels=["a", "b"],
    )
    input_ids = [sample["input_ids"] for sample in concat_dataset]

    assert len(concat_dataset) == 7
    assert all(isinstance(i, (int, np.integer)) for i in input_ids)
    assert set(sample["dataset_name"] for sample in concat_dataset) <= {"a", "b"}
    assert all("idx" in sample and "dataset_idx" in sample for sample in concat_dataset)

    dataset_names = [sample["dataset_name"] for sample in concat_dataset]
    assert "a" in dataset_names
    assert "b" in dataset_names
