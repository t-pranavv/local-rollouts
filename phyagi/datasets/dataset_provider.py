# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from torch.utils.data import Dataset

from phyagi.utils.file_utils import get_full_path
from phyagi.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetProviderConfig:
    """Dataset provider configuration (used as default for language modeling datasets).

    Args:
        cache_dir: Path to the cached dataset directory.
        train_file: Path to the cached ``.npy`` train file.
        validation_file: Path to the cached ``.npy`` validation file.
        validation_split: Part of the training dataset to be used for dynamic validation dataset creation.
            If type is ``int``, will use number of tokens from the training dataset.
            If type is ``float``, will use fraction of the training dataset.
        tokenizer_file: Path to the cached ``.pkl`` tokenizer file.
        seq_len: Sequence length.
        shift_labels: Whether to shift labels.
        weight: Weight for the dataset.
        label: Label for the dataset.
        ignore_token_id: Index to be ignored in the labels.
        seed: Random seed.
        random_mask_prob: Probability of applying a mask that ignores random samples (sequence lengths).
            If ``None``, ignores masking.

    """

    cache_dir: Optional[Union[str, Path]] = field(
        default=None, metadata={"help": "Path to the cached dataset directory."}
    )

    train_file: Optional[Union[str, Path]] = field(
        default=None, metadata={"help": "Path to the cached `.npy` train file."}
    )

    validation_file: Optional[Union[str, Path]] = field(
        default=None, metadata={"help": "Path to the cached `.npy` validation file."}
    )

    validation_split: Optional[Union[int, float]] = field(
        default=None, metadata={"help": "Part of the training dataset to be used for validation."}
    )

    tokenizer_file: Optional[str] = field(default=None, metadata={"help": "Path to the cached `.pkl` tokenizer file."})

    seq_len: int = field(default=2048, metadata={"help": "Sequence length."})

    shift_labels: bool = field(default=False, metadata={"help": "Whether to shift labels."})

    weight: float = field(default=1.0, metadata={"help": "Weight for the dataset."})

    label: Optional[str] = field(default=None, metadata={"help": "Label for the dataset."})

    ignore_token_id: int = field(default=-100, metadata={"help": "Index to be ignored in the labels."})

    seed: int = field(default=42, metadata={"help": "Random seed."})

    random_mask_prob: Optional[float] = field(
        default=None,
        metadata={"help": "Probability of applying a mask that ignores random samples (sequence lengths)."},
    )

    def __post_init__(self) -> None:
        if self.cache_dir:
            if self.train_file is not None or self.validation_file is not None:
                raise ValueError(
                    f"`cache_dir` and, `train_file` or `validation_file` can not be specified at the same time, but got '{self.cache_dir}' and, '{self.train_file}' or '{self.validation_file}'."
                )

            # If `cache_dir` is provided, we gather its complete path
            self.cache_dir = get_full_path(self.cache_dir)
            if not self.cache_dir.exists():
                raise FileNotFoundError(f"`cache_dir` must be a valid path, but got '{self.cache_dir}'.")

            # If `cache_dir` is a directory, we set default names for the files
            if self.cache_dir.is_dir():
                self.train_file = self.cache_dir / "train.npy"
                self.validation_file = self.cache_dir / "validation.npy"
            # If `cache_dir` is a file, we assume the folder is for training only
            else:
                self.train_file = self.cache_dir
                self.validation_file = None

        # After `cache_dir` is processed, `train_file` and `validation_file` must be validated
        # If a file is not valid, warn the user and set it to None
        if self.train_file is not None:
            self.train_file = get_full_path(self.train_file)
            if not self.train_file.exists():
                logger.warning(f"`train_file` must be a valid path, but got '{self.train_file}'.")
                self.train_file = None

        if self.validation_split is not None:
            logger.warning("Setting `validation_file=None` since `validation_split` is provided.")
            self.validation_file = None

        if self.validation_file is not None:
            self.validation_file = get_full_path(self.validation_file)
            if not self.validation_file.exists():
                logger.warning(f"`validation_file` must be a valid path, but got '{self.validation_file}'.")
                self.validation_file = None

        if self.random_mask_prob is not None and not (0.0 <= self.random_mask_prob <= 1.0):
            raise ValueError(f"`random_mask_prob` must be in [0, 1], but got {self.random_mask_prob}.")
        if self.seq_len <= 0:
            raise ValueError(f"`seq_len` must be > 0, but got {self.seq_len}.")
        if self.weight <= 0.0:
            raise ValueError(f"`weight` must be > 0.0, but got {self.weight}.")

    @classmethod
    def from_dict(cls: DatasetProviderConfig, config_dict: Dict[str, Any]) -> DatasetProviderConfig:
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

        config = asdict(self)

        # `cache_dir` is only used to find the individual files
        config.pop("cache_dir", None)

        return config


class DatasetProvider:
    """Abstract class for dataset providers.

    This class serves as a base for implementing dataset providers that can return
    training and validation datasets. It enforces implementation of three methods:
    :meth:`get_train_dataset`, :meth:`get_val_dataset` and :meth:`get_tokenizer`.

    Examples:
        >>> class MyDatasetProvider(DatasetProvider):
        >>>     def __init__(self) -> None:
        >>>         pass
        >>>
        >>>     def get_train_dataset(self) -> Optional[Dataset]:
        >>>         return torchvision.datasets.MNIST(train=True)
        >>>
        >>>     def get_val_dataset(self) -> Optional[Dataset]:
        >>>         return torchvision.datasets.MNIST(train=False)
        >>>
        >>>     def get_tokenizer(self) -> Optional[Any]:
        >>>         return transformers.AutoTokenizer()

    """

    def __init__(self) -> None:
        """Initialize the dataset provider."""

        raise NotImplementedError("`DatasetProvider` must implement `__init__()`.")

    @classmethod
    def from_config(cls: DatasetProvider, config: DatasetProviderConfig) -> DatasetProvider:
        """Load a dataset provider from a configuration.

        Args:
            config: Dataset provider configuration.

        Returns:
            Dataset provider.

        """

        return cls(**config.to_dict())

    def get_train_dataset(self) -> Optional[Dataset]:
        """Get a training dataset.

        Returns:
            An instance of a training dataset.

        """

        raise NotImplementedError("`DatasetProvider` must implement `get_train_dataset()`.")

    def get_val_dataset(self) -> Optional[Dataset]:
        """Get a validation dataset.

        Returns:
            An instance of a validation dataset.

        """

        raise NotImplementedError("`DatasetProvider` must implement `get_val_dataset()`.")

    def get_tokenizer(self) -> Optional[Any]:
        """Get a tokenizer.

        Returns:
            An instance of a tokenizer.

        """

        raise NotImplementedError("`DatasetProvider` must implement `get_tokenizer()`.")
