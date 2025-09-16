# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class LMDataset(Dataset):
    """Language modeling dataset.

    This dataset is used for language modeling tasks. It takes a contiguous array
    of tokens and convert it into a dataset of sequences of length ``seq_len`` by
    sliding a window of size ``seq_len`` over the array of tokens.

    Each sequence is composed of ``input_ids`` and ``labels`` tensors.

    """

    def __init__(
        self,
        input_ids: np.array,
        labels: Optional[np.array] = None,
        seq_len: int = 2048,
        shift_labels: bool = False,
        ignore_token_id: int = -100,
        random_mask_prob: Optional[float] = None,
        random_mask_offset: int = 32768,
        seed: int = 42,
    ) -> None:
        """Initialize the dataset.

        Args:
            input_ids: Inputs array (encoded data).
            seq_len: Sequence length.
            labels: Labels array (encoded data).
                If ``None``, the labels will be inferred from the ``input_ids``.
            shift_labels: Whether labels must be shifted by one position.
            ignore_token_id: Index to be ignored in the labels.
            random_mask_prob: Probability of applying a mask that ignores random samples (sequence lengths).
                If ``None``, ignores masking.
            random_mask_offset: Value added to the token that will cause it to be masked.
            seed: Seed for the mask random number generator.

        """

        super().__init__()

        # `input_ids` and `labels` should not be sliced since they could be memory mapped
        self._input_ids = input_ids
        self._labels = labels
        self._label_offset = 1 if shift_labels else 0

        self._seq_len = seq_len
        self._n_input_ids = ((len(self._input_ids) - 1) // self._seq_len) * self._seq_len + 1
        self._n_sequences = math.ceil((self._n_input_ids - 1) / self._seq_len)

        self._ignore_token_id = ignore_token_id
        self._random_mask_prob = random_mask_prob
        self._random_mask_offset = random_mask_offset

        self._rng = np.random.default_rng(seed)

        if self._labels is not None and len(self._input_ids) != len(self._labels):
            raise ValueError(
                f"`input_ids` and `labels` must have the same size, but got {len(self._input_ids)} and {len(self._labels)}."
            )

    def __len__(self) -> int:
        return self._n_sequences

    @property
    def seq_len(self) -> int:
        """Sequence length of the dataset used for logging."""

        return self._seq_len

    def _apply_random_mask(
        self, input_ids: torch.LongTensor, labels: torch.LongTensor
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Apply a random mask to the ``input_ids`` and ``labels`` tensors.

        If a token is to be ignored, the value is modified by taking the modulo of the
        ``random_mask_offset``, which reduces it to a range of ``[0, random_mask_offset - 1]``.

        For ``input_ids`` we remove that value while for ``labels`` we mask the tokens that
        have that value and remove it.

        Args:
            input_ids: Input tensor.
            labels: Labels tensor.

        Returns:
            Tuple with the updated ``input_ids`` and ``labels`` tensors.

        """

        input_ids = input_ids % self._random_mask_offset

        if self._rng.random() < self._random_mask_prob:
            mask = labels >= self._random_mask_offset
            labels = labels % self._random_mask_offset
            labels[mask] = self._ignore_token_id
        else:
            labels = labels % self._random_mask_offset

        return input_ids, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.LongTensor]:
        start = idx * self._seq_len
        end = min(start + self._seq_len, self._n_input_ids - 1)

        raw_labels = self._input_ids if self._labels is None else self._labels

        input_ids = torch.from_numpy(self._input_ids[start:end].astype(np.int64))
        labels = torch.from_numpy(raw_labels[start + self._label_offset : end + self._label_offset].astype(np.int64))

        if self._random_mask_prob is not None:
            input_ids, labels = self._apply_random_mask(input_ids, labels)

        return {"input_ids": input_ids, "labels": labels}
