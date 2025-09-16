# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from unittest.mock import patch

import pytest

from phyagi.utils.download_utils import download_dataset, download_tokenizer


@patch("subprocess.run")
@patch.dict(os.environ, {"BLOB_SAS_TOKEN": "test_token"})
@pytest.mark.is_mpi
def test_download_dataset(mock_subprocess_run):
    dataset_configs = [{"cache_dir": "/tmp/data/train.npy"}]

    download_dataset(
        dataset_configs,
        storage_account="testaccount",
        container="testcontainer",
        data_root="/tmp/data",
        local_data_root="/tmp/local",
    )

    mock_subprocess_run.assert_called_once_with(
        'azcopy copy "https://testaccount.blob.core.windows.net/testcontainer/train.npy?test_token" "/tmp/local" --include-path "train.npy" ',
        shell=True,
    )


@patch("subprocess.run")
@patch.dict(os.environ, {"BLOB_SAS_TOKEN": "test_token"})
@pytest.mark.is_mpi
def test_download_tokenizer(mock_subprocess_run):
    download_tokenizer(
        storage_account="testaccount",
        container="testcontainer",
        tokenizer_path="/tokenizer/",
        local_data_root="/tmp/local",
    )

    mock_subprocess_run.assert_called_once_with(
        'azcopy copy "https://testaccount.blob.core.windows.net/testcontainer//tokenizer/?test_token" /tmp/local --recursive',
        shell=True,
    )
