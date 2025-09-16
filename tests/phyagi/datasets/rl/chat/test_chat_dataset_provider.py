# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from phyagi.datasets.rl.chat.chat_dataset_provider import (
    ChatDatasetProvider,
    ChatDatasetProviderConfig,
)


def test_chat_dataset_provider():
    dp = ChatDatasetProvider(train_file="train.parquet", validation_file="validation.parquet")
    assert dp._train_file == "train.parquet"
    assert dp._validation_file == "validation.parquet"


def test_chat_dataset_provider_from_config():
    config = ChatDatasetProviderConfig(train_file="train.parquet")
    dp = ChatDatasetProvider.from_config(config)

    # `train_file` and `validation_file` are set to None if not available
    assert dp._train_file is None
    assert dp._validation_file is None
    assert dp._tokenizer is None
    assert dp._messages_column_name == "question"
    assert dp._ground_truth_column_name == "answer"
    assert dp._max_length == 1024
    assert dp._filter_max_length is False
    assert dp._tokenize is True
    assert dp.weight == 1.0
