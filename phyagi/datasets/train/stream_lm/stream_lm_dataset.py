# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Dict, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset as TorchIterableDataset


class StreamLMDataset(TorchIterableDataset):
    """Stream-based language modeling dataset.

    Each element is a fixed-length sequence of pre-tokenized token identifiers, while the dataset
    is designed to be lazy and shard-aware, optimising for memory and I/O efficiency.

    - Slices by rank: each data-parallel worker sees a disjoint slice of the dataset.
    - Prefetches contiguous chunks: maximises sequential disk I/O.
    - Keeps no duplicates in RAM: the buffer holds at most ``read_ahead`` rows.

    It is assumed that the labels are the same as the input ids (causal language modeling),
    and therefore the ``shift_labels`` option is not supported. Additionally, the backing NumPy array
    must be C-contiguous; otherwise, reshaping a mem-mapped file would trigger a full copy.

    """

    def __init__(
        self,
        dataset_name: str,
        input_ids: np.array,
        labels: Optional[np.array] = None,
        seq_len: int = 2048,
        shift_labels: bool = False,
        read_ahead: int = 2048,
        chunk_size: int = 1024,
    ) -> None:
        """Initialize the dataset.

        Args:
            dataset_name: Unique name for the dataset.
            input_ids: 2D or 1D NumPy array (mem-mapped or in-memory) of token ids.
                If 2D, its second dimension must equal ``seq_len``.
            labels: Optional 2D or 1D NumPy array (mem-mapped or in-memory) of labels.
                If provided, it must have the same shape as ``input_ids``.
            seq_len: Length of each sequence (number of tokens).
            shift_labels: Whether to shift the labels by one token to the right.
            read_ahead: Minimum number of prefetched rows kept in the buffers used to controls the
                trade-off between I/O throughput and RAM usage.
            chunk_size: Number of contiguous rows read in one refill call.
                It must be ``>= read_ahead`` to avoid double reads.

        """

        super().__init__()

        if shift_labels:
            raise ValueError("`StreamLMDataset` does not support `shift_labels`.")
        if not input_ids.flags["C_CONTIGUOUS"]:
            raise RuntimeError("`input_ids` must be c-contiguous for streaming.")
        if labels is not None and labels.size != input_ids.size:
            raise RuntimeError("`labels` must have the same size as `input_ids`.")

        self._name = dataset_name
        self._input_ids = input_ids
        self._seq_len = seq_len
        self._chunk_size = chunk_size
        self._read_ahead = read_ahead
        self._dataset_len = self._input_ids.size // self._seq_len

        self._input_ids = self._input_ids.reshape(-1)[: self._dataset_len * self._seq_len]
        self._input_ids = np.reshape(self._input_ids, (self._dataset_len, self._seq_len))

        self._labels = labels
        if self._labels is not None:
            self._labels = self._labels.reshape(-1)[: self._dataset_len * self._seq_len]
            self._labels = np.reshape(self._labels, (self._dataset_len, self._seq_len))

        # Mutable state attributes
        self._ptr = None
        self._input_ids_buffer = []
        self._labels_buffer = []
        self._shard_size = None
        self._start = None
        self._end = None
        self._length = None

    def __iter__(self) -> Iterator[Dict[str, torch.LongTensor]]:
        while True:
            while not self._input_ids_buffer:
                self._refill()

            while len(self._input_ids_buffer) < self._read_ahead:
                self._refill()

            input_ids = torch.tensor(self._input_ids_buffer.pop(0).copy(), dtype=torch.long)
            labels = input_ids
            if self._labels is not None:
                labels = torch.tensor(self._labels_buffer.pop(0).copy(), dtype=torch.long)

            yield {"input_ids": input_ids, "labels": labels}

    def __len__(self) -> int:
        return self._dataset_len

    @property
    def seq_len(self) -> int:
        """Sequence length of the dataset used for logging."""

        return self._seq_len

    def _refill(self) -> None:
        if self._ptr >= self._end:
            # If we reached the end of the shard, reset the pointer
            self._ptr = self._start

        stop = min(self._ptr + self._chunk_size, self._end)

        self._input_ids_buffer.extend(self._input_ids[self._ptr : stop])
        if self._labels is not None:
            self._labels_buffer.extend(self._labels[self._ptr : stop])

        self._ptr = stop

    def set_rank_and_world_size(self, rank: Optional[int], world_size: Optional[int]) -> None:
        """Set the rank and world size for distributed training.

        Args:
            rank: Local rank in the data-parallel job (0 <= rank < world_size).
            world_size: Total number of data-parallel workers.

        """

        if rank is None or world_size is None:
            raise ValueError("`rank` and `world_size` must be provided to `set_rank_and_world_size`.")

        self._shard_size = int(np.ceil(self._dataset_len / world_size))
        self._start = rank * self._shard_size
        self._ptr = self._start
        self._end = min(self._start + self._shard_size, self._dataset_len)
        self._length = self._end - self._start

    def state_dict(self) -> Dict[str, int]:
        """State of the dataset as a dictionary.

        Returns:
            Dictionary containing the current step count and the state of each dataset.

        """

        return {
            "name": self._name,
            "ptr": self._ptr,
            "pending": len(self._input_ids_buffer),
        }

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        """Load the state of the dataset from a dictionary.

        Args:
            state: Dictionary containing the state of the dataset.

        """

        if self._start is None:
            raise RuntimeError("`set_rank_and_world_size` needs to be called before `load_state_dict`.")
        if self._name != state_dict["name"]:
            raise RuntimeError(
                f"Dataset names must match when loading states, but got {self._name} and {state_dict['name']}."
            )

        ptr = state_dict["ptr"] - state_dict["pending"]
        ptr_rel = (ptr - self._start) % self._length  # Wrap whithin the shard
        self._ptr = self._start + ptr_rel

        # _refill() will populate the buffers again
        self._input_ids_buffer = []
        self._labels_buffer = []
