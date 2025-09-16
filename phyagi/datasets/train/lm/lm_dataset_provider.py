# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pickle
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from phyagi.datasets.dataset_provider import DatasetProvider
from phyagi.datasets.shared_memory_utils import (
    process_dataset_to_memory,
    save_memory_dataset,
)
from phyagi.datasets.train.lm.lm_dataset import LMDataset
from phyagi.utils.file_utils import (
    is_file_available,
    save_json_file,
    validate_file_extension,
)
from phyagi.utils.logging_utils import get_logger
from phyagi.utils.type_utils import xor

logger = get_logger(__name__)


def _create_val_split(
    dataset_dict: DatasetDict,
    val_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> DatasetDict:
    key_to_split, key_to_assign = ("train", "validation")
    are_keys_valid = key_to_split in dataset_dict and key_to_assign not in dataset_dict

    if are_keys_valid and val_split and val_split > 0.0:
        tmp_dataset_dict = dataset_dict[key_to_split].train_test_split(test_size=val_split, shuffle=shuffle, seed=seed)

        dataset_dict[key_to_split] = tmp_dataset_dict["train"]
        dataset_dict[key_to_assign] = tmp_dataset_dict["test"]

    return dataset_dict


def _split_ndarray(
    array: np.ndarray, axis: int = -1, split: Union[int, float] = 0.0, shuffle: bool = False, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    if array.ndim < 1:
        raise ValueError(f"`array.ndim` must be >= 1, but got {array.ndim}.")
    length = array.shape[axis]

    split = split or 0.0
    n_idx = int(split * length) if isinstance(split, float) else split
    if not (0 <= n_idx <= length):
        raise ValueError(
            f"`n_idx` must be in [0, {length}], but got {n_idx}. `split` needs to be adjusted to a valid value."
        )

    if shuffle:
        # Set the random seed, which is used to persist shuffled indexes across multiple calls
        rng = np.random.RandomState(seed)
        idx = rng.permutation(length)

        selected_array = np.take(array, idx[:n_idx], axis=axis)
        remaining_array = np.take(array, idx[n_idx:], axis=axis)
    else:
        selected_array, remaining_array = np.split(array, [n_idx], axis=axis)

    return remaining_array, selected_array


def _tokenize_concatenated(
    examples: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizerBase = None,
    mapping_column_name: List[str] = None,
    use_eos_token: bool = False,
    dtype: Optional[np.dtype] = None,
) -> Dict[str, Any]:
    if tokenizer is None:
        raise ValueError("`tokenizer` must be provided, but got None.")
    if mapping_column_name is None:
        raise ValueError("`mapping_column_name` must be provided, but got None.")

    def _add_eos_token(examples: List[str]) -> List[str]:
        return [example + tokenizer.eos_token if example else example for example in examples]

    examples_mapping = tuple(
        _add_eos_token(examples[column_name]) if use_eos_token else examples[column_name]
        for column_name in mapping_column_name
    )

    tokenized_examples = tokenizer(*examples_mapping, truncation=False, padding=False)
    concat_tokenized_examples = np.fromiter(chain(*tokenized_examples["input_ids"]), dtype=dtype)

    return {"input_ids": [concat_tokenized_examples], "length": [len(concat_tokenized_examples)]}


def _tokenize_dataset(
    dataset_dict: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    mapping_fn: Callable[[Any], Dict[str, Any]],
    mapping_fn_kwargs: Dict[str, Any],
    mapping_column_name: Union[str, List[str]],
    use_eos_token: bool,
    dtype: np.dtype,
    num_workers: int,
) -> DatasetDict:
    mapping_column_name = mapping_column_name if isinstance(mapping_column_name, list) else [mapping_column_name]
    mapping_fn = mapping_fn or _tokenize_concatenated
    mapping_fn_kwargs = mapping_fn_kwargs or {
        "tokenizer": tokenizer,
        "mapping_column_name": mapping_column_name,
        "use_eos_token": use_eos_token,
        "dtype": dtype,
    }

    return dataset_dict.map(
        mapping_fn,
        fn_kwargs=mapping_fn_kwargs,
        batched=True,
        num_proc=num_workers,
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenizing dataset...",
    )


class LMDatasetProvider(DatasetProvider):
    """Language modeling dataset provider."""

    def __init__(
        self,
        train_file: Optional[Union[str, Path]] = None,
        validation_file: Optional[Union[str, Path]] = None,
        validation_split: Optional[Union[int, float]] = None,
        tokenizer_file: Optional[Union[str, Path]] = None,
        seq_len: int = 2048,
        shift_labels: bool = False,
        weight: float = 1.0,
        label: Optional[str] = None,
        ignore_token_id: int = -100,
        random_mask_prob: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize the dataset provider.

        Args:
            train_file: Path to the ``.npy`` training array file.
            validation_file: Path to the ``.npy`` validation array file.
            validation_split: Part of the training dataset to be used for dynamic validation dataset creation.
                If type is ``int``, will use number of tokens from the training dataset.
                If type is ``float``, will use fraction of the training dataset.
            tokenizer_file: Path to the ``.pkl`` tokenizer file.
            seq_len: Sequence length.
            shift_labels: Whether to shift labels.
            weight: Weight for the dataset.
            label: Label for the dataset.
            ignore_token_id: Index to be ignored in the labels.
            random_mask_prob: Probability of applying a mask that ignores random samples (sequence lengths).
                If ``None``, ignores masking.
            seed: Seed for the mask random number generator.

        """

        # `input_ids` attributes are used to avoid loading the files multiple times
        self._train_input_ids, self._train_labels = None, None
        self._validation_input_ids, self._validation_labels = None, None

        self._train_file = train_file
        self._validation_file = validation_file
        self._validation_split = validation_split
        self._tokenizer_file = tokenizer_file
        self._seq_len = seq_len
        self._shift_labels = shift_labels
        self._weight = weight
        self._label = label
        self._ignore_token_id = ignore_token_id
        self._random_mask_prob = random_mask_prob
        self._seed = seed

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def label(self) -> Optional[str]:
        return self._label

    @classmethod
    def from_hub(
        cls: LMDatasetProvider,
        dataset_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        disk_file_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[List[str], Dict[str, Union[str, List[str]]]]] = None,
        revision: Optional[str] = None,
        tokenizer: Union[PreTrainedTokenizerBase, str] = None,
        mapping_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
        mapping_fn_kwargs: Optional[Dict[str, Any]] = None,
        mapping_column_name: Optional[Union[str, List[str]]] = None,
        validation_split: float = 0.0,
        shuffle: bool = True,
        seed: int = 42,
        num_workers: int = 1,
        use_eos_token: bool = True,
        use_shared_memory: bool = True,
        cache_dir: Optional[Union[str, Path]] = "cache",
        raw_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> LMDatasetProvider:
        """Load a dataset provider by downloading and tokenizing data from Hugging Face hub.

        Args:
            dataset_path: Path of the dataset when loading with ``load_dataset()``.
            dataset_name: Name of the dataset when loading with ``load_dataset()``.
            disk_file_path: Path of the disk file when loading with ``load_from_disk()``.
                If different than ``None``, will override dataset downloaded from Hugging Face hub.
            data_dir: Path to the data directory.
            data_files: Path to the source data files.
            revision: Version of the dataset.
            tokenizer: Instance of tokenizer to use.
                If type is ``str``, will use :class:`transformers.AutoTokenizer` to load the tokenizer.
            mapping_fn: A function that maps the dataset.
                If set to ``None``, :func:`phyagi.datasets.train.lm.lm_dataset_provider._tokenize_concatenated` will be used.
            mapping_fn_kwargs: Keyword arguments to be passed to ``mapping_fn``.
            mapping_column_name: Columns in the dataset to be tokenized.
                If type is ``str``, only one column will be tokenized.
                If type is ``List[str]``, multiple columns will be tokenized.
            validation_split: Fraction of the dataset to be used for validation.
                If set to ``0.0``, it will not create additional validation split.
            shuffle: Whether to shuffle the dataset.
            seed: Random seed.
            num_workers: Number of workers to use for tokenizing.
            use_eos_token: Whether to use the EOS token to separate sequences.
            use_shared_memory: Whether to use the shared memory for caching.
            cache_dir: Output directory where dataset will be cached.
            raw_dir: Output directory where raw dataset will be saved.
                If set to ``None``, the raw dataset will not be saved.

        Returns:
            Dataset provider.

        """

        if not xor(dataset_path, disk_file_path):
            raise ValueError("`dataset_path` and `disk_file_path` are mutually exclusive.")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError(f"`tokenizer` must be an instance of PreTrainedTokenizerBase, but got '{type(tokenizer)}'.")

        mapping_column_name = mapping_column_name or ["text"]
        processing_column_name = ["input_ids"]
        dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32

        logger.info("Loading non-tokenized dataset...")
        if disk_file_path:
            dataset_dict = load_from_disk(disk_file_path)
        else:
            dataset_dict = load_dataset(
                dataset_path,
                name=dataset_name,
                data_dir=data_dir,
                data_files=data_files,
                revision=revision,
            )

        # `cache_dir` must be created after loading the dataset since a user can provide a relative path
        # and it will break when trying to load the dataset from that path
        if cache_dir is None:
            raise ValueError("`cache_dir` must be provided, but got None.")
        cache_dir = Path(cache_dir)
        if cache_dir.is_dir():
            logger.warning(f"'{cache_dir}' already exists and will be overritten.")
        cache_dir.mkdir(parents=True, exist_ok=True)

        if raw_dir is not None:
            raw_dir = Path(raw_dir)
            if raw_dir.is_dir():
                logger.warning(f"'{raw_dir}' already exists and will be overritten.")
            dataset_dict.save_to_disk(raw_dir)

        # Ensure dataset is always a dictionary with at least `train` split since the
        # following functions expect it to be a dictionary
        if not isinstance(dataset_dict, DatasetDict):
            dataset_dict = DatasetDict({"train": dataset_dict})

        logger.info("Creating validation split (if necessary)...")
        dataset_dict = _create_val_split(dataset_dict, val_split=validation_split, shuffle=shuffle, seed=seed)

        tokenized_dataset_dict = _tokenize_dataset(
            dataset_dict,
            tokenizer,
            mapping_fn,
            mapping_fn_kwargs,
            mapping_column_name,
            use_eos_token,
            dtype,
            num_workers,
        )
        processed_dataset_dict = process_dataset_to_memory(
            tokenized_dataset_dict,
            cache_dir,
            dtype,
            processing_column_name,
            num_workers=num_workers,
            use_shared_memory=use_shared_memory,
        )

        logger.info(f"Saving tokenized dataset: {cache_dir}")
        cache_files = save_memory_dataset(
            processed_dataset_dict, tokenizer, cache_dir, use_shared_memory=use_shared_memory
        )

        config = {
            "dataset_path": dataset_path,
            "dataset_name": dataset_name,
            "disk_file_path": disk_file_path,
            "data_dir": data_dir,
            "data_files": data_files,
            "revision": revision,
            "tokenizer": {
                "name_or_path": tokenizer.name_or_path,
                "model_max_length": tokenizer.model_max_length,
            },
            "mapping_column_name": mapping_column_name,
            "shuffle": shuffle,
            "seed": seed,
            "num_workers": num_workers,
            "use_eos_token": use_eos_token,
            "use_shared_memory": use_shared_memory,
        }
        save_json_file(config, cache_dir / "config.json")

        return cls(**cache_files, **kwargs)

    @classmethod
    def from_cache(cls: LMDatasetProvider, cache_dir: Union[str, Path], **kwargs) -> LMDatasetProvider:
        """Load a dataset provider from a cache directory.

        If a cached file (``train.npy``, ``validation.npy`` or ``tokenizer.pkl``)
        is missing, it will be set to ``None``.

        Args:
            cache_dir: Cached dataset directory.

        Returns:
            Dataset provider.

        """

        cache_dir = Path(cache_dir)
        cache_files = {}

        for split in ["train", "validation"]:
            cache_file = cache_dir / f"{split}.npy"
            cache_files[f"{split}_file"] = cache_file if cache_file.is_file() else None

        tokenizer_file = cache_dir / "tokenizer.pkl"
        cache_files["tokenizer_file"] = tokenizer_file if tokenizer_file.is_file() else None

        return cls(**cache_files, **kwargs)

    def get_train_dataset(self) -> Optional[LMDataset]:
        if validate_file_extension(self._train_file, ".npy"):
            self._train_input_ids = np.load(self._train_file, mmap_mode="r")
            if self._train_input_ids.ndim == 2:
                self._train_input_ids = self._train_input_ids.squeeze()

            labels_file = is_file_available(self._train_file, file_extension="_labels.npy")
            if labels_file is not None:
                self._train_labels = np.load(str(labels_file), mmap_mode="r")

            type_info = np.iinfo(self._train_input_ids.dtype)
            random_mask_offset = (type_info.max + 1) // 2

            # If we are creating a dynamic validation file, we need to ensure that
            # the `train_input_ids` do not overlap with the `validation_input_ids`
            if isinstance(self._validation_split, (int, float)):
                self._train_input_ids, self._validation_input_ids = _split_ndarray(
                    self._train_input_ids, split=self._validation_split
                )
                if self._train_labels is not None:
                    self._train_labels, self._validation_labels = _split_ndarray(
                        self._train_labels, split=self._validation_split
                    )

            return LMDataset(
                self._train_input_ids,
                labels=self._train_labels,
                seq_len=self._seq_len,
                shift_labels=self._shift_labels,
                ignore_token_id=self._ignore_token_id,
                random_mask_prob=self._random_mask_prob,
                random_mask_offset=random_mask_offset,
                seed=self._seed,
            )

        return None

    def get_val_dataset(self) -> Optional[LMDataset]:
        if validate_file_extension(self._validation_file, ".npy"):
            self._validation_input_ids = np.load(self._validation_file, mmap_mode="r")
            if self._validation_input_ids.ndim == 2:
                self._validation_input_ids = self._validation_input_ids.squeeze()

            labels_file = is_file_available(self._validation_file, file_extension="_labels.npy")
            if labels_file is not None:
                self._validation_labels = np.load(str(labels_file), mmap_mode="r")

        if self._validation_input_ids is not None:
            type_info = np.iinfo(self._validation_input_ids.dtype)
            random_mask_offset = (type_info.max + 1) // 2

            return LMDataset(
                self._validation_input_ids,
                labels=self._validation_labels,
                seq_len=self._seq_len,
                shift_labels=self._shift_labels,
                ignore_token_id=self._ignore_token_id,
                random_mask_prob=self._random_mask_prob,
                random_mask_offset=random_mask_offset,
                seed=self._seed,
            )

        return None

    def get_tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        if validate_file_extension(self._tokenizer_file, ".pkl"):
            with open(self._tokenizer_file, "rb") as f:
                return pickle.load(f)

        return None
