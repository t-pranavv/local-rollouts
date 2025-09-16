# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil

from transformers import PreTrainedTokenizerBase

from phyagi.datasets.dataset_provider import DatasetProviderConfig
from phyagi.datasets.train.lm.lm_dataset import LMDataset
from phyagi.datasets.train.lm.lm_dataset_provider import LMDatasetProvider


def test_lm_dataset_provider():
    dp = LMDatasetProvider(train_file="train.npy", validation_file="validation.npy")
    assert dp._train_file == "train.npy"
    assert dp._validation_file == "validation.npy"
    assert dp._tokenizer_file is None


def test_lm_dataset_provider_from_config():
    config = DatasetProviderConfig(
        train_file="train.npy",
        validation_file="validation.npy",
        tokenizer_file="tokenizer.pkl",
        seq_len=1024,
        shift_labels=True,
        weight=0.5,
    )
    dp = LMDatasetProvider.from_config(config)

    # `train_file` and `validation_file` are set to None if not available
    assert dp._train_file is None
    assert dp._validation_file is None
    assert dp._tokenizer_file == "tokenizer.pkl"
    assert dp._seq_len == 1024
    assert dp._shift_labels is True
    assert dp.weight == 0.5


def test_lm_dataset_provider_from_hub():
    text_file = "tmp.txt"
    with open(text_file, "w") as f:
        for _ in range(100):
            f.write("This is a test\n")

    dp = LMDatasetProvider.from_hub(
        dataset_path="text",
        data_files={"train": text_file},
        tokenizer="microsoft/phi-2",
        validation_split=0.1,
        cache_dir="cache",
    )
    os.remove(text_file)

    train_dataset = dp.get_train_dataset()
    val_dataset = dp.get_val_dataset()
    tokenizer = dp.get_tokenizer()

    assert isinstance(train_dataset, LMDataset)
    assert isinstance(val_dataset, LMDataset)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)


def test_lm_dataset_provider_from_cache():
    dp = LMDatasetProvider.from_cache("cache")

    train_dataset = dp.get_train_dataset()
    val_dataset = dp.get_val_dataset()
    tokenizer = dp.get_tokenizer()

    assert isinstance(train_dataset, LMDataset)
    assert isinstance(val_dataset, LMDataset)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    shutil.rmtree("cache")
