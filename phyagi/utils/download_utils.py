# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Union


def download_dataset(
    dataset_configs: Union[Dict[str, Any], List[Dict[str, Any]]],
    storage_account: str,
    container: str,
    data_root: Union[str, Path] = "/tmp/data",
    local_data_root: Union[str, Path] = "/tmp/data",
    cap_mbps: int = None,
) -> None:
    """Download datasets using ``azcopy``.

    Args:
        dataset_configs: Dataset configuration.
        storage_account: Name of the blob storage account.
        container: Name of the blob storage container.
        data_root: Root directory of the blob storage datasets.
        local_data_root: Local root directory for downloading datasets.
        cap_mbps: Maximum download speed in MBps.

    """

    data_root = Path(data_root)
    local_data_root = Path(local_data_root)

    # Azure often provides a token without a leading `?`, so we add it if it is missing
    if "BLOB_SAS_TOKEN" not in os.environ:
        raise EnvironmentError("`BLOB_SAS_TOKEN` must be set as an environment variable.")
    sas_token = os.environ["BLOB_SAS_TOKEN"]
    sas_token = "?" + sas_token if sas_token[0] != "?" else sas_token

    ALLOWED_KEYS = ["cache_dir", "dataset_path", "train_file", "validation_file"]

    dataset_configs = [dataset_configs] if not isinstance(dataset_configs, list) else dataset_configs
    for dc in dataset_configs:
        for key, value in dc.items():
            # Only download files that are explicitly specified in the `ALLOWED_KEYS`
            if key not in ALLOWED_KEYS:
                continue

            # Ensure that file's value is always a string to prevent errors
            if not isinstance(value, str):
                continue

            path_without_root = value.replace(str(data_root), "")
            source_path_split = path_without_root.split("/")

            # If dataset path does not contain filename (either explit name or using wildcard),
            # assume default filename
            if source_path_split[-1].find(".npy") == -1 and source_path_split[-1].find(".tsv") == -1:
                filename = "train.npy"
                source_path = "/".join(source_path_split)[1:]
                target_path = local_data_root / "/".join(source_path_split[:-1])[1:]
            else:
                filename = source_path_split[-1]
                source_path = "/".join(source_path_split[:-1])[1:]
                target_path = local_data_root / "/".join(source_path_split[:-2])[1:]

            target_path = str(target_path).rstrip("/")

            cap_str = ""
            if cap_mbps is not None:
                cap_str = f"--cap-mbps {cap_mbps}"

            # Download specific file (requires SAS token with READ permission)
            if filename.find("*") == -1 and filename[-4:] in [".npy", ".tsv"]:
                command = 'azcopy copy "https://{}.blob.core.windows.net/{}/{}{}" "{}" --include-path "{}" {}'.format(
                    storage_account, container, source_path + filename, sas_token, target_path, filename, cap_str
                )

            # Download all files in directory or based on matching pattern
            # (requires SAS token with READ & LIST permissions)
            else:
                command = 'azcopy copy "https://{}.blob.core.windows.net/{}/{}/{}" "{}" --include-pattern "{}" {} --recursive'.format(
                    storage_account, container, source_path, sas_token, target_path, filename, cap_str
                )

            subprocess.run(command, shell=True)


def download_tokenizer(
    storage_account: str,
    container: str,
    tokenizer_path: str,
    local_data_root: Union[str, Path] = "/tmp/data",
) -> None:
    """Download a tokenizer using ``azcopy``.

    Args:
        storage_account: Name of the blob storage account.
        container: Name of the blob storage container.
        tokenizer_path: Path to the tokenizer files.
        local_data_root: Local root directory for downloading tokenizers.

    """

    # Azure often provides a token without a leading `?`, so we add it if it is missing
    if "BLOB_SAS_TOKEN" not in os.environ:
        raise EnvironmentError("`BLOB_SAS_TOKEN` must be set as an environment variable.")
    sas_token = os.environ["BLOB_SAS_TOKEN"]
    sas_token = "?" + sas_token if sas_token[0] != "?" else sas_token

    source_path = tokenizer_path + sas_token
    command = 'azcopy copy "https://{}.blob.core.windows.net/{}/{}" {} --recursive'.format(
        storage_account, container, source_path, str(local_data_root)
    )

    subprocess.run(command, shell=True)
