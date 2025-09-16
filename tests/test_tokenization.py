# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import shutil

from transformers import PreTrainedTokenizerBase

from phyagi.datasets.train.lm.lm_dataset_provider import LMDatasetProvider


def test_lm_tokenization():
    dataset_provider = LMDatasetProvider.from_hub(
        dataset_path="glue",
        dataset_name="sst2",
        tokenizer="microsoft/phi-1",
        mapping_column_name="sentence",
        validation_split=0.1,
        shuffle=True,
        seed=42,
        num_workers=1,
        use_eos_token=True,
        use_shared_memory=True,
        cache_dir="test_tmp",
        seq_len=512,
    )

    train_dataset = dataset_provider.get_train_dataset()
    assert train_dataset[0]["input_ids"].shape == (512,)
    assert train_dataset[0]["labels"].shape == (512,)

    val_dataset = dataset_provider.get_val_dataset()
    assert val_dataset[0]["input_ids"].shape == (512,)
    assert val_dataset[0]["labels"].shape == (512,)

    tokenizer = dataset_provider.get_tokenizer()
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    shutil.rmtree("test_tmp")
