# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from datasets import Dataset as HfDataset
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from phyagi.datasets.dataset_provider import DatasetProvider
from phyagi.datasets.rl.chat.chat_dataset import ChatDataset
from phyagi.utils.file_utils import get_full_path, validate_file_extension
from phyagi.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_tokenizer(tokenizer: Union[PreTrainedTokenizerBase, str, Dict[str, Any]]) -> PreTrainedTokenizerBase:
    if isinstance(tokenizer, str):
        return AutoTokenizer.from_pretrained(tokenizer)
    if isinstance(tokenizer, dict):
        return AutoTokenizer.from_pretrained(**tokenizer)
    return tokenizer


@dataclass
class ChatDatasetProviderConfig:
    """Chat dataset provider configuration.

    Args:
        train_file: Path to the training file.
        validation_file: Path to the validation file.
        tokenizer: Instance of tokenizer to use.
            If type is ``str``, will use :class:`transformers.AutoTokenizer` to load the tokenizer.
            If type is ``dict``, will use :class:`transformers.AutoTokenizer` and keyword arguments to load the tokenizer.
        messages_column_name: Column name for the messages.
        ground_truth_column_name: Column name for the ground truth.
        max_length: Maximum sequence length.
        filter_max_length: Whether it should filter sequences longer than ``max_length`` during pre-tokenization.
        tokenize: Whether it should pre-tokenize the data.
        weight: Weight for the dataset.
        label: Label for the dataset.
        subsample: Fraction of the dataset to use.
            If ``None``, no subsampling is applied.

    """

    train_file: Optional[Union[str, Path]] = field(default=None, metadata={"help": "Path to the training file."})

    validation_file: Optional[Union[str, Path]] = field(default=None, metadata={"help": "Path to the validation file."})

    tokenizer: Union[PreTrainedTokenizerBase, str, Dict[str, Any]] = field(
        default=None, metadata={"help": "Instance of tokenizer to use."}
    )

    messages_column_name: str = field(default="question", metadata={"help": "Column name for the messages."})

    ground_truth_column_name: str = field(default="answer", metadata={"help": "Column name for the ground truth."})

    max_length: Optional[int] = field(default=1024, metadata={"help": "Maximum sequence length."})

    filter_max_length: bool = field(
        default=False,
        metadata={"help": "Whether it should filter sequences longer than `max_length` during pre-tokenization."},
    )

    tokenize: bool = field(default=True, metadata={"help": "Whether it should pre-tokenize the data."})

    weight: float = field(default=1.0, metadata={"help": "Weight for the dataset."})

    label: Optional[str] = field(default=None, metadata={"help": "Label for the dataset."})

    subsample: Optional[float] = field(default=None, metadata={"help": "Fraction of the dataset to use."})

    def __post_init__(self) -> None:
        if self.train_file is not None:
            self.train_file = get_full_path(self.train_file)
            if not self.train_file.exists():
                logger.warning(f"`train_file` must be a valid path, but got '{self.train_file}'.")
                self.train_file = None

        if self.validation_file is not None:
            self.validation_file = get_full_path(self.validation_file)
            if not self.validation_file.exists():
                logger.warning(f"`validation_file` must be a valid path, but got '{self.validation_file}'.")
                self.validation_file = None

        if self.weight <= 0.0:
            raise ValueError(f"`weight` must be > 0.0, but got {self.weight}.")

    @classmethod
    def from_dict(cls: ChatDatasetProviderConfig, config_dict: Dict[str, Any]) -> ChatDatasetProviderConfig:
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


class ChatDatasetProvider(DatasetProvider):
    """Chat dataset provider."""

    def __init__(
        self,
        train_file: Optional[Union[str, Path]] = None,
        validation_file: Optional[Union[str, Path]] = None,
        tokenizer: Optional[Union[PreTrainedTokenizerBase, str, Dict[str, Any]]] = None,
        messages_column_name: str = "messages",
        ground_truth_column_name: str = "ground_truth",
        max_length: Optional[int] = None,
        filter_max_length: bool = False,
        tokenize: bool = True,
        weight: float = 1.0,
        label: Optional[str] = None,
        subsample: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Initialize the dataset provider.

        Args:
            train_file: Path to the training file.
            validation_file: Path to the validation file.
            tokenizer: Instance of tokenizer to use.
                If type is ``str``, will use :class:`transformers.AutoTokenizer` to load the tokenizer.
                If type is ``dict``, will use :class:`transformers.AutoTokenizer` and keyword arguments to load the tokenizer.
            messages_column_name: Column name for the messages.
            ground_truth_column_name: Column name for the ground truth.
            max_length: Maximum sequence length.
            filter_max_length: Whether it should filter sequences longer than ``max_length`` during pre-tokenization.
            tokenize: Whether it should pre-tokenize the data.
            weight: Weight for the dataset.
            label: Label for the dataset.
            subsample: Fraction of the dataset to use.
                If ``None``, no subsampling is applied.

        """

        self._train_file = train_file
        self._validation_file = validation_file
        self._tokenizer = _load_tokenizer(tokenizer)
        self._messages_column_name = messages_column_name
        self._ground_truth_column_name = ground_truth_column_name
        self._max_length = max_length
        self._filter_max_length = filter_max_length
        self._tokenize = tokenize
        self._weight = weight
        self._label = label
        self._subsample = subsample or 1.0

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def label(self) -> Optional[str]:
        return self._label

    def _prepare_dataset(self, file_path: str) -> HfDataset:
        dataset = load_dataset("parquet", data_files=file_path)
        if self._subsample < 1.0:
            for split in dataset.keys():
                n_rows = int(len(dataset[split]) * self._subsample)
                dataset[split] = dataset[split].shuffle(seed=1).select(range(n_rows))
        return dataset

    def get_train_dataset(self) -> Optional[ChatDataset]:
        if validate_file_extension(self._train_file, ".parquet"):
            train_dataset = ChatDataset(
                self._prepare_dataset(str(self._train_file)),
                tokenizer=self._tokenizer,
                messages_column_name=self._messages_column_name,
                ground_truth_column_name=self._ground_truth_column_name,
                max_length=self._max_length,
                filter_max_length=self._filter_max_length,
                tokenize=self._tokenize,
            )
            train_dataset.assert_within_max_length()

            if self._filter_max_length:
                logger.info(
                    f"Removed {train_dataset._removed_rows} samples from {self._train_file} since they exceeded the maximum length of {self._max_length} tokens."
                )
            return train_dataset

        return None

    def get_val_dataset(self) -> Optional[ChatDataset]:
        if validate_file_extension(self._validation_file, ".parquet"):
            val_dataset = ChatDataset(
                self._prepare_dataset(str(self._validation_file)),
                tokenizer=self._tokenizer,
                messages_column_name=self._messages_column_name,
                ground_truth_column_name=self._ground_truth_column_name,
                max_length=self._max_length,
                filter_max_length=self._filter_max_length,
                tokenize=self._tokenize,
            )
            val_dataset.assert_within_max_length()

            if self._filter_max_length:
                logger.critical(
                    f"Removed {val_dataset._removed_rows} samples from {self._validation_file} since they exceeded the maximum length of {self._max_length} tokens. Be aware that this could change your validation results!"
                )
            return val_dataset

        return None

    def get_tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        return self._tokenizer
