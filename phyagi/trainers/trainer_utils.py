# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import math
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler


class BatchTracker:
    """Batch tracker to keep track of the sample indices for each dataset."""

    def __init__(self) -> None:
        """Initialize the batch tracker."""

        self._tracker = {}

    @property
    def samples_idx_per_dataset(self) -> Dict[int, List[int]]:
        """Get the sample indices for each dataset."""

        samples_idx_per_dataset = sorted(self._tracker.items(), key=lambda x: x[0])
        return {k: v for k, v in samples_idx_per_dataset}

    @property
    def n_samples_per_dataset(self) -> Dict[int, int]:
        """Get the number of samples for each dataset."""

        samples_idx_per_dataset = sorted(self._tracker.items(), key=lambda x: x[0])
        return {k: len(v) for k, v in samples_idx_per_dataset}

    def update(self, batch: Dict[str, torch.Tensor]) -> None:
        """Update the batch tracker with the sample indices for each dataset.

        Args:
            batch: Batch of data.

        """

        if not isinstance(batch, dict):
            return

        sample_idx = batch.get("idx", None)
        dataset_idx = batch.get("dataset_idx", None)
        if sample_idx is None or dataset_idx is None:
            return

        sample_idx = sample_idx.tolist()
        dataset_idx = dataset_idx.tolist()
        if len(sample_idx) != len(dataset_idx):
            raise ValueError(
                f"`sample_idx` and `dataset_idx` must have the same length, but got {len(sample_idx)} and {len(dataset_idx)}."
            )

        for d_idx, s_idx in zip(dataset_idx, sample_idx):
            if d_idx not in self._tracker:
                self._tracker[d_idx] = []
            self._tracker[d_idx].append(s_idx)

    def reset(self) -> None:
        """Reset the batch tracker."""

        self._tracker = {}


class RepeatingLoader:
    """Repeating data loader."""

    def __init__(self, loader: Iterator, use_batch_tracker: bool = False) -> None:
        """Wrap an iterator to allow for infinite iteration.

        This is especially useful for DataLoader types that we wish to automatically
        restart upon completion. This version supports shuffling the sampler between
        epochs.

        Args:
            loader: Data loader to repeat.
            use_batch_tracker: Whether to use the batch tracker to keep track of the
                sample indices for each dataset (useful for debugging).

        """

        self.loader = loader
        self.data_iter = iter(self.loader)
        self.batch_tracker = BatchTracker() if use_batch_tracker else None

    def __iter__(self) -> RepeatingLoader:
        return self

    def __next__(self) -> Tuple[torch.Tensor, ...]:
        try:
            batch = next(self.data_iter)
        except StopIteration:
            # If the loader has a sampler and it is shuffling, then we need to
            # increment the epoch counter
            if getattr(self.loader, "sampler", None) is not None:
                if getattr(self.loader.sampler, "shuffle", False):
                    self.loader.sampler.set_epoch(self.loader.sampler.epoch + 1)

            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)

        if self.batch_tracker is not None:
            self.batch_tracker.update(batch)

        return batch


class _DistributedSamplerWithNumpyRNG(DistributedSampler):
    def __iter__(self) -> Iterator:
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            rng = np.random.default_rng(seed=self.seed + self.epoch)
            indices = rng.permutation(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[: self.total_size]
        if len(indices) != self.total_size:
            raise ValueError(f"`indices` must have a length of {self.total_size}, but got {len(indices)}.")

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise ValueError(f"`indices` must have a length of {self.num_samples}, but got {len(indices)}.")

        return iter(indices)


class StatefulDistributedSampler(_DistributedSamplerWithNumpyRNG):
    """Distributed sampler that supports resuming from a given step.

    This class uses Numpy's random number generator instead of :class:`torch.utils.data.DistributedSampler`.
    
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
        epoch: int = 0,
        total_consumed_samples: int = 0,
    ) -> None:
        """Stateful distributed sampler.

        Args:
            dataset: PyTorch dataset.
            num_replicas: Number of processes participating in distributed training.
                By default, ``world_size`` is retrieved from the current distributed group.
            rank: Rank of the current process within ``num_replicas``.
                By default, ``rank`` is retrieved from the current distributed group.
            shuffle: If ``True`` (default), sampler will shuffle the indices.
            seed: Random seed used to shuffle the sampler if ``shuffle=True``.
                This number should be identical across all processes in the distributed group.
            drop_last: If ``True``, then the sampler will drop the tail of the data to make it
                evenly divisible across the number of replicas. If ``False``, the sampler will
                add extra indices to make the data evenly divisible across the replicas.
            epoch: Epoch to start from.
            total_consumed_samples: Number of samples consumed to start from.

        """

        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)

        self.epoch = epoch
        self.total_consumed_samples = total_consumed_samples

    def __iter__(self) -> Iterator:
        indices = list(super().__iter__())
        return iter(indices[((self.total_consumed_samples // self.num_replicas) % self.num_samples) :])
