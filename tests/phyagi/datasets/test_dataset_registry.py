# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import numpy as np
import pytest

from phyagi.datasets.concat_dataset import WeightedConcatDataset
from phyagi.datasets.registry import get_data_collator, get_dataset
from phyagi.datasets.train.train_data_collator import LMDataCollator


def test_get_data_collator():
    collator = get_data_collator(data_collator_type="lm")
    assert isinstance(collator, LMDataCollator)

    with pytest.raises(ValueError) as excinfo:
        get_data_collator(data_collator_type="invalid_type")
    assert isinstance(excinfo.value, ValueError)


def test_get_dataset():
    dataset_file = "temp_dataset.npy"
    dataset = np.ones((20480), dtype=np.int32)
    np.save(dataset_file, dataset)

    dataset_config = {"train_file": dataset_file}
    train_dataset, eval_dataset = get_dataset(dataset_config)
    assert isinstance(train_dataset, WeightedConcatDataset)
    assert eval_dataset is None

    os.remove(dataset_file)
