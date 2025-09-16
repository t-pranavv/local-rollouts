# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import mmap
import pickle
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase


class _SHMArray(np.ndarray):
    def __new__(cls: _SHMArray, input_array: np.ndarray, shm: Optional[SharedMemory] = None) -> _SHMArray:
        obj = np.asarray(input_array).view(cls)
        obj.shm = shm

        return obj

    def __array_finalize__(self, obj: _SHMArray) -> None:
        if obj is None:
            return

        self.shm = getattr(obj, "shm", None)


def _process_with_shared_memory(
    dataset_dict: DatasetDict,
    dtype: np.dtype,
    processing_column_name: List[str],
    num_workers: int = 1,
) -> Dict[str, _SHMArray]:
    def __process_with_shared_memory(example: Dict[str, Any], name: str, columns: List[str], length: int) -> None:
        shared_memory = SharedMemory(name=name)
        n_columns = len(columns)

        shared_memory_array = np.ndarray((n_columns, length), dtype=dtype, buffer=shared_memory.buf)
        for i, column in enumerate(columns):
            start_idx = example["offset"] - len(example[column])
            shared_memory_array[i, start_idx : example["offset"]] = example[column]

        shared_memory.close()

    processed_dataset_dict = {}
    for name, ds in dataset_dict.items():
        dataset_dict[name] = ds.add_column("offset", np.cumsum(ds["length"]))

        n_columns = len(processing_column_name)
        length = dataset_dict[name][-1]["offset"]

        shared_memory = SharedMemory(create=True, size=n_columns * length * np.dtype(dtype).itemsize)
        shared_memory_name = shared_memory.name

        dataset_dict[name].map(
            __process_with_shared_memory,
            fn_kwargs={"name": shared_memory_name, "columns": processing_column_name, "length": length},
            batched=False,
            num_proc=num_workers,
            desc="Processing dataset with shared memory...",
        )

        shared_memory_array = np.ndarray((n_columns, length), dtype=dtype, buffer=shared_memory.buf)
        processed_dataset_dict[name] = _SHMArray(shared_memory_array, shm=shared_memory)

    return processed_dataset_dict


def _process_with_memory_map_files(
    dataset_dict: DatasetDict,
    cache_dir: Path,
    dtype: np.dtype,
    processing_column_name: List[str],
    num_workers: int = 1,
) -> Dict[str, np.ndarray]:
    def __process_with_memory_map_files(
        example: Dict[str, Any], file_path: Path, columns: List[str], length: int
    ) -> None:
        with open(file_path, "r+b") as f:
            memory_map = mmap.mmap(f.fileno(), 0)
            n_columns = len(columns)

            memory_map_array = np.ndarray(
                (
                    n_columns,
                    length,
                ),
                dtype=dtype,
                buffer=memory_map,
            )
            for i, column in enumerate(columns):
                start_idx = example["offset"] - len(example[column])
                memory_map_array[i, start_idx : example["offset"]] = example[column]

            memory_map.flush()

    processed_dataset_dict = {}
    for split, dataset in dataset_dict.items():
        dataset_dict[split] = dataset.add_column("offset", np.cumsum(dataset["length"]))

        length = dataset_dict[split][-1]["offset"]
        n_columns = len(processing_column_name)

        file_path = cache_dir / f"{split}.bin"
        with open(file_path.as_posix(), "wb") as f:
            f.truncate(n_columns * length * np.dtype(dtype).itemsize)

        dataset_dict[split].map(
            __process_with_memory_map_files,
            fn_kwargs={"file_path": file_path, "columns": processing_column_name, "length": length},
            batched=False,
            num_proc=num_workers,
            desc="Processing dataset with memory map files...",
        )

        processed_dataset_dict[split] = np.memmap(file_path, dtype=dtype, mode="r", shape=(n_columns, length))

    return processed_dataset_dict


def process_dataset_to_memory(
    dataset_dict: DatasetDict,
    cache_dir: Path,
    dtype: np.dtype,
    processing_column_name: List[str],
    num_workers: int = 1,
    use_shared_memory: bool = True,
) -> Dict[str, Union[_SHMArray, np.ndarray]]:
    """Process the dataset to memory.

    Args:
        dataset_dict: Dataset dictionary.
        cache_dir: Cache directory.
        dtype: NumPy data type.
        processing_column_name: Name of the column(s) to use for processing.
        num_workers: Number of workers.
        use_shared_memory: Whether to use shared memory.

    Returns:
        Dictionary with memory-processed datasets.

    """

    if use_shared_memory:
        return _process_with_shared_memory(dataset_dict, dtype, processing_column_name, num_workers=num_workers)

    return _process_with_memory_map_files(
        dataset_dict, cache_dir, dtype, processing_column_name, num_workers=num_workers
    )


def save_memory_dataset(
    dataset_dict: Dict[str, Union[_SHMArray, np.ndarray]],
    tokenizer: PreTrainedTokenizerBase,
    cache_dir: Union[str, Path],
    use_shared_memory: bool = True,
) -> Dict[str, Path]:
    """Save a memory-processed dataset to a cache directory.

    Args:
        dataset_dict: Memory-processed dataset dictionary.
        tokenizer: Tokenizer used to encode the data.
        cache_dir: Cache directory.
        use_shared_memory: Whether shared memory has been used to process the dataset.

    Returns:
        Dictionary with paths to the saved files.

    """

    cache_dir = Path(cache_dir)
    cache_files = {}

    for split, dataset in dataset_dict.items():
        npy_file_path = cache_dir / f"{split}.npy"
        np.save(npy_file_path, dataset)

        # If using shared memory, dataset needs to have its shared memory
        # unlinked to prevent memory leak
        if use_shared_memory:
            dataset.shm.unlink()

        # If not using shared memory, dataset needs to have its memory map
        # closed to prevent an additional .bin file
        if not use_shared_memory:
            dataset._mmap.close()
            Path(cache_dir / f"{split}.bin").unlink()

        cache_files[f"{split}_file"] = npy_file_path

    # Save tokenizer to a pickle file and record its path
    tokenizer_file_path = cache_dir / "tokenizer.pkl"
    with open(tokenizer_file_path, "wb") as f:
        pickle.dump(tokenizer, f)
    cache_files["tokenizer_file"] = tokenizer_file_path

    return cache_files
