# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.utils.data import Dataset

from phyagi.trainers.trainer_utils import (
    BatchTracker,
    RepeatingLoader,
    StatefulDistributedSampler,
)


class MockSampler:
    def __init__(self, shuffle, epoch):
        self.shuffle = shuffle
        self.epoch = epoch

    def set_epoch(self, epoch):
        self.epoch = epoch


class MockLoader:
    def __init__(self, data, shuffle=False):
        self.data = data
        self.epoch = 0
        self.shuffle = shuffle
        self.sampler = MockSampler(shuffle=self.shuffle, epoch=self.epoch)

    def __iter__(self):
        self.data_iter = iter(self.data)
        return self.data_iter

    def set_epoch(self, epoch):
        self.epoch = epoch


def test_batch_tracker():
    bt = BatchTracker()
    assert bt._tracker == {}

    batch = {"idx": torch.tensor([1, 2]), "dataset_idx": torch.tensor([0, 1])}
    bt.update(batch)
    assert bt._tracker == {0: [1], 1: [2]}
    assert bt.samples_idx_per_dataset == {0: [1], 1: [2]}
    assert bt.n_samples_per_dataset == {0: 1, 1: 1}

    bt.reset()
    assert bt._tracker == {}


def test_repeating_loader():
    data = [(torch.tensor([1]), torch.tensor([2])), (torch.tensor([3]), torch.tensor([4]))]

    mock_loader = MockLoader(data)
    repeating_loader = RepeatingLoader(mock_loader)
    for _ in range(4):
        batch = next(repeating_loader)
    assert batch == (torch.tensor([3]), torch.tensor([4]))

    mock_loader_with_shuffle = MockLoader(data, shuffle=True)
    repeating_loader_with_shuffle = RepeatingLoader(mock_loader_with_shuffle)
    for _ in range(4):
        batch = next(repeating_loader_with_shuffle)
    assert repeating_loader_with_shuffle.loader.sampler.epoch == 1


class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.tensor([index], dtype=torch.float32)


def test_stateful_distributed_sampler():
    dataset = DummyDataset(100)
    sampler = StatefulDistributedSampler(dataset, num_replicas=2, rank=0, total_consumed_samples=20, shuffle=False)

    assert sampler.num_replicas == 2
    assert sampler.rank == 0
    assert sampler.total_consumed_samples == 20

    indices = list(iter(sampler))
    expected_indices = [i * 2 for i in range(10, 50)]
    assert indices == expected_indices

    dataset = DummyDataset(11)
    sampler = StatefulDistributedSampler(
        dataset, num_replicas=2, rank=0, total_consumed_samples=2, shuffle=True, seed=42, drop_last=True
    )

    assert sampler.num_replicas == 2
    assert sampler.rank == 0
    assert sampler.total_consumed_samples == 2

    indices = list(iter(sampler))

    expected_indices = [0, 3, 2, 9]
    assert indices == expected_indices

    dataset = DummyDataset(11)
    sampler = StatefulDistributedSampler(
        dataset, num_replicas=2, rank=0, total_consumed_samples=2, shuffle=True, seed=42, drop_last=False
    )

    assert sampler.num_replicas == 2
    assert sampler.rank == 0
    assert sampler.total_consumed_samples == 2

    indices = list(iter(sampler))

    expected_indices = [0, 3, 2, 9, 8]
    assert indices == expected_indices
