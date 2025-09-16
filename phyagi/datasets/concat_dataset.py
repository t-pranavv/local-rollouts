# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import bisect
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, IterableDataset, Subset

from phyagi.utils.file_utils import load_json_file, save_json_file
from phyagi.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _advance_rng(rng: np.random.Generator, n: int) -> None:
    if n <= 0:
        return
    try:
        rng.bit_generator.advance(n)
    except AttributeError:  # older versions of NumPy might burn numbers
        rng.random(n)


def _gather_local_states(state_obj: Any, world_size: int) -> List[Any]:
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, state_obj)
    return gathered


class WeightedConcatDataset(ConcatDataset):
    """Weighted concatenation of datasets."""

    def __init__(self, datasets: Iterable[Dataset], weights: Iterable[float], **kwargs) -> None:
        """Initialize the dataset by concatenating the given datasets according
        to the given weights.

        For a given dataset, the number of resulting samples is computed as
        ``ceil(len(dataset) * weight)``. If the resulting number of samples is
        greater than the length of the dataset, the dataset is concatenated
        multiple times to reach the desired number of samples.

        Args:
            datasets: List of datasets.
            weights: List of weights.

        """

        datasets = list(datasets)
        weights = list(weights)
        if len(datasets) == 0 or len(weights) == 0:
            raise ValueError("`datasets` or `weights` cannot be empty.")
        if len(datasets) != len(weights):
            raise ValueError(
                f"`datasets` and `weights` must have the same length, but got {len(datasets)} and {len(weights)}."
            )

        weighted_datasets = []
        for dataset, weight in zip(datasets, weights):
            n_samples = math.ceil(len(dataset) * weight)

            if n_samples > len(dataset):
                n_concats = n_samples // len(dataset)
                n_remainder = n_samples % len(dataset)

                samples_idx = np.concatenate(
                    [np.concatenate([np.arange(len(dataset))] * n_concats), np.arange(n_remainder)]
                ).tolist()
            else:
                samples_idx = np.arange(n_samples).tolist()

            subset = Subset(dataset, samples_idx)
            subset.seq_len = getattr(dataset, "seq_len", 1)

            weighted_datasets.append(subset)

        super().__init__(weighted_datasets)

    @property
    def seq_len(self) -> int:
        """Sequence length of the dataset used for logging."""

        return self.datasets[0].seq_len

    def __getitem__(self, idx: int) -> Any:
        if idx < 0:
            if -idx > len(self):
                raise IndexError(f"{idx} out of range for dataset with size of {len(self)}.")
            idx = len(self) + idx

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        sample = self.datasets[dataset_idx][sample_idx]
        if isinstance(sample, dict):
            sample["idx"] = idx
            sample["dataset_idx"] = dataset_idx

        return sample


class SequentialWeightedConcatDataset(WeightedConcatDataset):
    """Sequential weighted concatenation of datasets."""

    def __init__(self, datasets: Iterable[Dataset], weights: Iterable[float], **kwargs) -> None:
        """Initialize the dataset by sequentially concatenating the given datasets according
        to the given weights.

        The concatenating order is determined by the order of the given datasets. If a given
        dataset has a weight greater than 1, it will be sequentially concatenated multiple times.

        Examples:
            >>> datasets = [A, B, C]
            >>> weights = [1.5, 0.3, 0.2]
            >>> dataset = SequentialWeightedConcatDataset(datasets, weights)
            >>> assert len(dataset[:len(A) * 1.5]) == len(A)

        Args:
            datasets: List of datasets.
            weights: List of weights.

        """

        super().__init__(datasets, weights)

        self._dataset_idx = 0
        self._consumed_samples = 0
        self._cumulative_markers = [np.zeros(size) for size in self.cumulative_sizes]

    def __getitem__(self, idx: int) -> Any:
        if idx < 0:
            if -idx > len(self):
                raise IndexError(f"{idx} out of range for dataset with size of {len(self)}.")
            idx = len(self) + idx

        # When `consumed_samples` is greater than the length of current dataset,
        # we need to reset it and move to next dataset
        if self._consumed_samples == self.cumulative_sizes[self._dataset_idx]:
            self._dataset_idx += 1

        if idx >= self.cumulative_sizes[self._dataset_idx]:
            # When `idx` is equal or bigger than current dataset size,
            # we need to sample a valid index from current dataset
            sample_idx = np.random.choice(np.argwhere(self._cumulative_markers[self._dataset_idx] == 0)[0], 1)[0]
        else:
            # When `idx` is smaller than current dataset size,
            # we need to find the corresponding index in current dataset
            if self._dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[self._dataset_idx - 1]

        self._cumulative_markers[self._dataset_idx][sample_idx] = 1
        self._consumed_samples += 1

        sample = self.datasets[self._dataset_idx][sample_idx]
        if isinstance(sample, dict):
            sample["idx"] = idx
            sample["dataset_idx"] = self._dataset_idx

        return sample


class WeightedConcatIterableDataset(IterableDataset):
    """Weighted concatenation of multiple iterable datasets.

    Each step we draw a uniform random number, pick the dataset whose cumulative distribution function (CDF)
    interval contains it, and yield **one** sample from that dataset.

    - No in-RAM shuffling: each underlying file is assumed pre-shuffled.
    - Per-worker random number generator (RNG): distinct GPUs (and dataloader workers).
    - Tiny checkpoints: we store only a few integers per shard.

    """

    def __init__(self, datasets: Iterable[IterableDataset], weights: Iterable[float], **kwargs) -> None:
        """Initialize the dataset by concatenating the given datasets according
        to the given weights.

        Args:
            datasets: Iterable of already-constructed :class:`phyagi.datasets.train.stream_lm.stream_lm_dataset.StreamLMDataset`.
            weights: Relative sampling probabilities, which are normalized to sum to 1.

        """

        super().__init__()

        if len(datasets) != len(weights):
            raise ValueError("`datasets` and `weights` must have the same length.")

        self._datasets = datasets
        self._datasets_iter = [iter(dataset) for dataset in datasets]

        self._weights = np.array(weights, dtype=np.float64) * np.array(
            [len(dataset) for dataset in datasets], dtype=np.float64
        )
        self._weights /= self._weights.sum()
        self._cdf = self._weights.cumsum()

        # Independent (per worker) variables for iterating and recording the state
        self._rng = np.random.default_rng()
        self._steps = 0
        self._global_worker_id = 0
        self._rank = None
        self._world_size = None

    def __len__(self) -> int:
        return int(self._weights.sum())

    def __iter__(self) -> Iterable[Any]:
        while True:
            r = self._rng.random()
            self._steps += 1
            dataset_idx = int(np.searchsorted(self._cdf, r, side="right"))

            sample = next(self._datasets_iter[dataset_idx])
            if isinstance(sample, dict):
                sample["idx"] = -1
                sample["dataset_idx"] = dataset_idx

            yield sample

    @property
    def seq_len(self) -> int:
        """Sequence length of the dataset used for logging."""

        return getattr(self._datasets[0], "seq_len", 1)

    def set_rank_and_world_size(self, rank: Optional[int], world_size: Optional[int]) -> None:
        """Set the rank and world size for distributed training.

        Args:
            rank: Rank of the current process.
            world_size: Number of processes in the world.

        """

        if not torch.distributed.is_initialized():
            return

        rank = rank if rank is not None else torch.distributed.get_rank()
        world_size = world_size if world_size is not None else torch.distributed.get_world_size()

        for dataset in self._datasets:
            dataset.set_rank_and_world_size(rank, world_size)

        self._global_worker_id = rank
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            self._global_worker_id = rank * worker_info.num_workers + worker_info.id

        self._rng = np.random.default_rng(self._global_worker_id)

        self._rank = rank
        self._world_size = world_size

    def state_dict(self) -> Dict[str, Any]:
        """State of the dataset as a dictionary.

        Returns:
            Dictionary containing the current step count and the state of each dataset.

        """

        return {
            "steps": self._steps,
            "datasets": [d.state_dict() for d in self._datasets],
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state of the dataset from a dictionary.

        Args:
            state: Dictionary containing the state of the dataset.

        """

        self._steps = state_dict["steps"]

        for dataset, sd in zip(self._datasets, state_dict["datasets"]):
            dataset.load_state_dict(sd)

        _advance_rng(self._rng, self._steps)

    def save_state(self, output_dir: Union[str, Path], step: Union[str, int]) -> None:
        """Save the state of the datasets.

        Args:
            output_dir: Output directory.
            step: Step number.

        """

        local_state = self.state_dict()
        all_states = _gather_local_states(local_state, self._world_size)

        if self._rank == 0:
            file_path = Path(output_dir) / "dataset_state" / str(step) / "state.json"
            file_path.parent.mkdir(parents=True, exist_ok=True)

            full_state = {f"rank_{i}": sd for i, sd in enumerate(all_states)}
            save_json_file(full_state, file_path)

        torch.distributed.barrier()

    def load_state(self, load_dir: Union[str, Path], step: Optional[str] = None) -> None:
        """Load the state of the datasets.

        Args:
            load_dir: Load directory.
            step: Step number.

        """

        load_path = Path(load_dir)
        if step is None:
            latest_path = load_path / "latest"
            if latest_path.is_file():
                step = latest_path.read_text().strip()
            else:
                logger.warning(f"'{latest_path}' is not a valid path.")
                return

        file_path = load_path / "dataset_state" / str(step) / "state.json"
        try:
            state = load_json_file(file_path)
            self.load_state_dict(state[f"rank_{self._rank}"])
        except FileNotFoundError:
            logger.warning(f"Failed to load dataset state from '{file_path}'.")


class WeightedConcatChatDataset(WeightedConcatDataset):
    """Weighted concatenation of chat datasets."""

    def __init__(
        self, datasets: Iterable[Dataset], weights: Iterable[float], labels: Optional[Iterable[str]] = None
    ) -> None:
        """Initialize the dataset by concatenating the given chat datasets according
        to the given weights.

        Args:
            datasets: Chat-based datasets.
            weights: Weights for each dataset.
            labels: Labels for each dataset.
                If not provided, labels will be generated as "0", "1", "2", etc.
                If only one dataset is provided, the label will be ``None``.

        """

        super().__init__(datasets, weights)

        self._datasets = datasets
        self._weights = weights
        self._labels = (
            [None]
            if labels is None and len(datasets) == 1
            else [str(i) for i in range(len(datasets))]
            if labels is None
            else labels
        )

    def __repr__(self) -> str:
        return f"WeightedConcatChatDataset(datasets={[str(d) for d in self._datasets]}, weights={self._weights}, labels={self._labels})"

    def __getitem__(self, idx: int) -> Any:
        sample = super().__getitem__(idx)
        if isinstance(sample, dict):
            sample["dataset_name"] = self._labels[sample["dataset_idx"]]

        return sample
