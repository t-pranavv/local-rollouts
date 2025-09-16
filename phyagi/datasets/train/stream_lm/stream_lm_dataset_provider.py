# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pathlib as pl
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from phyagi.datasets.dataset_provider import DatasetProvider
from phyagi.datasets.train.stream_lm.stream_lm_dataset import StreamLMDataset
from phyagi.utils.file_utils import is_file_available, validate_file_extension
from phyagi.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class StreamLMDatasetProviderConfig:
    """Stream (non-tokenized) language modeling dataset provider configuration.

    Args:
        dataset_path: Path of the dataset when loading with :meth:`datasets.load_dataset`.
        seq_len: Sequence length.
        shift_labels: Whether to shift labels.
        weight: Weight for the dataset.
        label: Label for the dataset.
        seed: Random seed.

    """

    dataset_path: str = field(metadata={"help": "Path of the dataset, 1D or 2D npy array."})

    seq_len: int = field(default=2048, metadata={"help": "Sequence length."})

    shift_labels: bool = field(default=False, metadata={"help": "Whether to shift labels."})

    weight: float = field(default=1.0, metadata={"help": "Weight for the dataset."})

    label: Optional[str] = field(default=None, metadata={"help": "Label for the dataset."})

    seed: int = field(default=42, metadata={"help": "Random seed."})

    def __post_init__(self) -> None:
        self.label = self.dataset_path if self.label is None else self.label

        if self.seq_len <= 0:
            raise ValueError(f"`seq_len` must be > 0, but got {self.seq_len}.")
        if self.weight <= 0.0:
            raise ValueError(f"`weight` must be > 0.0, but got {self.weight}.")

    @classmethod
    def from_dict(cls: StreamLMDatasetProviderConfig, config_dict: Dict[str, Any]) -> StreamLMDatasetProviderConfig:
        """Load a configuration from a dictionary.

        Args:
            config_dict: Dictionary containing the configuration.

        Returns:
            Configuration.

        """

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.

        Returns:
            Dictionary containing the configuration.

        """

        return asdict(self)


class StreamLMDatasetProvider(DatasetProvider):
    """Stream (non-tokenized) language modeling dataset provider."""

    def __init__(
        self,
        dataset_path: str,
        seq_len: int = 2048,
        shift_labels: bool = False,
        weight: float = 1.0,
        label: Optional[str] = None,
        seed: int = 42,
        **kwargs,
    ) -> None:
        """Initialize the dataset provider.

        Args:
            dataset_path: Path of the dataset when loading with :meth:`datasets.load_dataset`.
            seq_len: Sequence length.
            shift_labels: Whether to shift labels.
            weight: Weight for the dataset.
            label: Label for the dataset.
            seed: Random seed.

        """

        self._dataset_path = dataset_path
        self._seq_len = seq_len
        self._shift_labels = shift_labels
        self._weight = weight
        self._label = label
        self._seed = seed

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def label(self) -> Optional[str]:
        return self._label

    def get_train_dataset(self) -> Optional[StreamLMDataset]:
        if not validate_file_extension(self._dataset_path, ".npy"):
            return None
        labels_file = is_file_available(self._dataset_path, file_extension="_labels.npy")

        input_ids = np.load(self._dataset_path, mmap_mode="r")
        labels = np.load(str(labels_file), mmap_mode="r") if labels_file is not None else None

        return StreamLMDataset(
            dataset_name="/".join(pl.Path(self._dataset_path).parts[-2:]),
            input_ids=input_ids,
            labels=labels,
            seq_len=self._seq_len,
            shift_labels=self._shift_labels,
        )

    def get_val_dataset(self) -> None:
        return None

    def get_tokenizer(self) -> None:
        return None
