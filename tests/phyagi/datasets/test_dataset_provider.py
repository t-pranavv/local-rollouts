# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock

import pytest

from phyagi.datasets.dataset_provider import DatasetProvider, DatasetProviderConfig


class MyDatasetProvider(DatasetProvider):
    def __init__(self):
        pass

    def get_train_dataset(self):
        return MagicMock()

    def get_val_dataset(self):
        return MagicMock()

    def get_tokenizer(self):
        return MagicMock()


def test_dataset_provider_config():
    config = DatasetProviderConfig(
        train_file="train.npy",
        validation_file="validation.npy",
        tokenizer_file="tokenizer.pkl",
        seq_len=1024,
        shift_labels=True,
        weight=0.5,
    )

    # `train_file` and `validation_file` are set to None if not available
    assert config.train_file is None
    assert config.validation_file is None

    assert config.tokenizer_file == "tokenizer.pkl"
    assert config.seq_len == 1024
    assert config.shift_labels is True
    assert config.weight == 0.5


def test_dataset_provider_config_conflicts():
    with pytest.raises(ValueError, match="cache_dir.*train_file"):
        DatasetProviderConfig(cache_dir="some/dir", train_file="train.npy")


def test_dataset_provider_config_invalid_attributes():
    with pytest.raises(ValueError, match="random_mask_prob.*in \[0, 1\]"):
        DatasetProviderConfig(random_mask_prob=1.5)

    with pytest.raises(ValueError, match="seq_len.*> 0"):
        DatasetProviderConfig(seq_len=0)

    with pytest.raises(ValueError, match="weight.*> 0.0"):
        DatasetProviderConfig(weight=0.0)


def test_my_dataset_provider():
    dataset_provider = MyDatasetProvider()

    train_dataset = dataset_provider.get_train_dataset()
    val_dataset = dataset_provider.get_val_dataset()
    tokenizer = dataset_provider.get_tokenizer()

    assert train_dataset is not None
    assert val_dataset is not None
    assert tokenizer is not None
    assert isinstance(train_dataset, MagicMock)
    assert isinstance(val_dataset, MagicMock)
    assert isinstance(tokenizer, MagicMock)
