# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile

import numpy as np
import pytest

from phyagi.datasets.train.stream_lm.stream_lm_dataset import StreamLMDataset
from phyagi.datasets.train.stream_lm.stream_lm_dataset_provider import (
    StreamLMDatasetProvider,
    StreamLMDatasetProviderConfig,
)


@pytest.fixture
def tmp_npy_file():
    fd, path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)

    data = np.arange(64, dtype=np.int32)
    np.save(path, data)

    yield path

    os.remove(path)


def test_stream_lm_dataset_provider(tmp_npy_file):
    dp = StreamLMDatasetProvider(tmp_npy_file, seq_len=8, shift_labels=False)
    assert dp._dataset_path == tmp_npy_file
    assert dp._seq_len == 8
    assert dp._shift_labels is False
    assert dp._seed == 42
    assert dp.weight == 1.0

    train_dataset = dp.get_train_dataset()
    val_dataset = dp.get_val_dataset()
    tokenizer = dp.get_tokenizer()
    assert isinstance(train_dataset, StreamLMDataset)
    assert val_dataset is None
    assert tokenizer is None


def test_stream_lm_dataset_provider_from_config(tmp_npy_file):
    config = StreamLMDatasetProviderConfig(
        dataset_path=tmp_npy_file,
        seq_len=16,
        shift_labels=False,
        weight=0.5,
        seed=123,
    )

    dp = StreamLMDatasetProvider.from_config(config)
    assert dp._dataset_path == tmp_npy_file
    assert dp._seq_len == 16
    assert dp._shift_labels is False
    assert dp._seed == 123
    assert dp.weight == 0.5

    train_dataset = dp.get_train_dataset()
    val_dataset = dp.get_val_dataset()
    tokenizer = dp.get_tokenizer()
    assert isinstance(train_dataset, StreamLMDataset)
    assert val_dataset is None
    assert tokenizer is None
