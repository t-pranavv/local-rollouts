# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from torch.utils.data import ConcatDataset, Dataset

from phyagi.datasets.concat_dataset import (
    SequentialWeightedConcatDataset,
    WeightedConcatChatDataset,
    WeightedConcatDataset,
    WeightedConcatIterableDataset,
)
from phyagi.datasets.dataset_provider import DatasetProvider, DatasetProviderConfig
from phyagi.datasets.rl.chat.chat_dataset_provider import (
    ChatDatasetProvider,
    ChatDatasetProviderConfig,
)
from phyagi.datasets.rl.rl_data_collator import RewardDataCollator
from phyagi.datasets.train.lm.lm_dataset_provider import LMDatasetProvider
from phyagi.datasets.train.stream_lm.stream_lm_dataset_provider import (
    StreamLMDatasetProvider,
    StreamLMDatasetProviderConfig,
)
from phyagi.datasets.train.train_data_collator import LMDataCollator
from phyagi.utils.config import filter_dict, filter_lists
from phyagi.utils.logging_utils import get_logger

logger = get_logger(__name__)

DATASET_PROVIDERS = {
    "lm": (DatasetProviderConfig, LMDatasetProvider),
    "stream_lm": (StreamLMDatasetProviderConfig, StreamLMDatasetProvider),
    "chat": (ChatDatasetProviderConfig, ChatDatasetProvider),
}

DATASET_CONCATS = {
    "random": WeightedConcatDataset,
    "random_iterable": WeightedConcatIterableDataset,
    "sequential": SequentialWeightedConcatDataset,
    "random_chat": WeightedConcatChatDataset,
}

DATA_COLLATORS = {
    "lm": LMDataCollator,
    "chat": RewardDataCollator,
}


def _get_dataset_provider(
    dataset_configs: List[Dict[str, Any]], dataset_provider: str = "lm"
) -> Tuple[List[Any], List[DatasetProvider]]:
    # Each `dataset_config` might have a different `dataset_provider`, but if it is not
    # provided, we use the global `dataset_provider`
    dataset_provider_config_cls, dataset_provider_cls = [], []
    for i, dc in enumerate(dataset_configs):
        dataset_provider_type = dc.get("dataset_provider", dataset_provider)
        if dataset_provider_type not in DATASET_PROVIDERS:
            raise ValueError(
                f"`dataset_provider` must be one of {list(DATASET_PROVIDERS.keys())}, but got '{dataset_provider_type}'."
            )

        # If `label` has not been provided in configuration, ensure it is set to a default value
        dc.update({"label": dc.get("label", str(i))})

        dataset_provider_config_cls.append(DATASET_PROVIDERS[dataset_provider_type][0])
        dataset_provider_cls.append(DATASET_PROVIDERS[dataset_provider_type][1])

    dataset_configs = [dpc.from_dict(dc) for dpc, dc in zip(dataset_provider_config_cls, dataset_configs)]
    dataset_providers = [dp.from_config(dc) for dp, dc in zip(dataset_provider_cls, dataset_configs)]

    return dataset_configs, dataset_providers


def _get_dataset_concat(
    datasets: List[Dataset],
    weights: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    dataset_concat: str = "random",
) -> ConcatDataset:
    if dataset_concat not in DATASET_CONCATS:
        raise ValueError(f"`dataset_concat` must be one of {list(DATASET_CONCATS.keys())}, but got '{dataset_concat}'.")

    dataset_concat_cls = DATASET_CONCATS[dataset_concat]
    weights = [1.0 for _ in datasets] if weights is None else weights

    return dataset_concat_cls(datasets, weights, labels=labels)


def get_dataset(
    dataset_configs: Union[Dict[str, Any], List[Dict[str, Any]]],
    dataset_concat: str = "random",
    eval_dataset_concat: Optional[str] = None,
    dataset_provider: str = "lm",
    dataset_global_weight: float = 1.0,
) -> Tuple[Dataset, Optional[Dict[str, Dataset]]]:
    """Get a tuple of training and evaluation datasets.

    Args:
        dataset_configs: Dataset configuration.
        dataset_concat: Dataset concatenation strategy (for training).
        eval_dataset_concat: Dataset concatenation strategy (for evaluation).
        dataset_provider: Dataset provider type.
            If any item from `dataset_configs` does not contain `dataset_provider`, it will be used as default.
        dataset_global_weight: Global weight multiplier applied to all datasets.

    Returns:
        Training and evaluation datasets.

    """

    logger.info("Loading datasets...")

    # Use the `dataset_configs` to create a list of dataset providers
    dataset_configs = [dataset_configs] if not isinstance(dataset_configs, list) else dataset_configs
    dataset_configs, dataset_providers = _get_dataset_provider(dataset_configs, dataset_provider=dataset_provider)
    dataset_labels = [dp.label for dp in dataset_providers]

    logger.info(f"Datasets: {dataset_configs}")
    logger.info(f"Global weight multiplier: {dataset_global_weight}")

    # Create the training dataset
    train_datasets = [dp.get_train_dataset() for dp in dataset_providers]
    train_weights = [dp.weight * dataset_global_weight for dp in dataset_providers]

    train_datasets, train_weights, train_labels = filter_lists(train_datasets, train_weights, dataset_labels)
    train_dataset = (
        _get_dataset_concat(train_datasets, weights=train_weights, labels=train_labels, dataset_concat=dataset_concat)
        if len(train_datasets) > 0
        else None
    )

    # Create the evaluation dataset (if available)
    eval_datasets = [dp.get_val_dataset() for dp in dataset_providers]

    # If `eval_dataset_concat` is specified, we filter and concatenate the datasets
    if eval_dataset_concat is not None:
        eval_datasets, eval_labels = filter_lists(eval_datasets, dataset_labels)
        eval_dataset = (
            {"0": _get_dataset_concat(eval_datasets, labels=eval_labels, dataset_concat=eval_dataset_concat)}
            if len(eval_datasets) > 0
            else None
        )

    # If `eval_dataset_concat` is not specified, we create a dictionary of datasets
    else:
        # Filter the dictionary to remove empty datasets, and check if there are any keys left
        eval_dataset = {dl: d for (dl, d) in zip(dataset_labels, eval_datasets)}
        eval_dataset = filter_dict(eval_dataset)
        eval_dataset = eval_dataset if len(eval_dataset) > 0 else None

    return train_dataset, eval_dataset


def get_data_collator(data_collator_type: str = "lm", **kwargs) -> Any:
    """Get a data collator.

    Args:
        data_collator_type: Data collator type.

    Returns:
        Data collator.

    """

    if data_collator_type not in DATA_COLLATORS:
        raise ValueError(
            f"`data_collator_type` must be one of {list(DATA_COLLATORS.keys())}, but got '{data_collator_type}'."
        )

    return DATA_COLLATORS[data_collator_type](**kwargs)
