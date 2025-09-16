# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import pytest
import torch

from phyagi.datasets.train.lm.lm_dataset import LMDataset


def test_lm_dataset():
    input_ids = np.arange(9, dtype=np.int32)
    seq_len = 2
    dataset = LMDataset(input_ids, seq_len=seq_len, shift_labels=False)

    assert len(dataset) == 4
    for i in range(len(dataset)):
        assert torch.equal(dataset[i]["input_ids"], torch.tensor(input_ids[i * seq_len : (i * seq_len) + seq_len]))
        assert torch.equal(dataset[i]["labels"], torch.tensor(input_ids[i * seq_len : (i * seq_len) + seq_len]))


def test_lm_dataset_with_shift():
    input_ids = np.arange(9, dtype=np.int32)
    seq_len = 2
    dataset = LMDataset(input_ids, seq_len=seq_len, shift_labels=True)

    assert len(dataset) == 4
    for i in range(len(dataset)):
        assert torch.equal(dataset[i]["input_ids"], torch.tensor(input_ids[i * seq_len : (i * seq_len) + seq_len]))
        assert torch.equal(dataset[i]["labels"], torch.tensor(input_ids[i * seq_len + 1 : (i * seq_len) + seq_len + 1]))


def test_lm_dataset_input_ids_less_than_seq_len():
    input_ids = np.arange(3)
    dataset = LMDataset(input_ids, seq_len=5)
    data = dataset[0]

    assert len(dataset) == 0
    assert data["input_ids"].shape[0] == 0


def test_lm_dataset_last_sequence_padding():
    input_ids = np.arange(10)
    seq_len = 3
    dataset = LMDataset(input_ids, seq_len=seq_len)
    last_data = dataset[len(dataset) - 1]

    assert last_data["input_ids"].shape[0] <= seq_len


def test_lm_dataset_mismatch_input_and_labels():
    input_ids = np.arange(10)
    labels = np.arange(8)

    with pytest.raises(ValueError):
        LMDataset(input_ids, labels=labels)


def test_lm_dataset_no_mask():
    input_ids = np.arange(9, dtype=np.int32)
    seq_len = 2
    type_info = np.iinfo(input_ids.dtype)

    random_mask_offset = (type_info.max + 1) // 2
    input_ids[1] += random_mask_offset
    input_ids[2] += random_mask_offset

    dataset = LMDataset(
        input_ids, seq_len=seq_len, shift_labels=False, random_mask_offset=random_mask_offset, random_mask_prob=None
    )

    assert dataset[0]["input_ids"][1] == input_ids[1]
    assert dataset[1]["input_ids"][0] == input_ids[2]


def test_lm_dataset_mask_zero_prob():
    input_ids = np.arange(9, dtype=np.int32)
    seq_len = 2
    type_info = np.iinfo(input_ids.dtype)

    random_mask_offset = (type_info.max + 1) // 2
    input_ids[1] += random_mask_offset
    input_ids[2] += random_mask_offset

    dataset = LMDataset(
        input_ids, seq_len=seq_len, shift_labels=False, random_mask_offset=random_mask_offset, random_mask_prob=0.0
    )

    assert dataset[0]["input_ids"][1] == input_ids[1] - random_mask_offset
    assert dataset[1]["input_ids"][0] == input_ids[2] - random_mask_offset
    assert dataset[0]["labels"][1] == input_ids[1] - random_mask_offset
    assert dataset[1]["labels"][0] == input_ids[2] - random_mask_offset


def test_lm_dataset_mask_full_prob():
    input_ids = np.arange(9, dtype=np.int32)
    seq_len = 2
    type_info = np.iinfo(input_ids.dtype)

    random_mask_offset = (type_info.max + 1) // 2
    input_ids[1] += random_mask_offset
    input_ids[2] += random_mask_offset

    dataset = LMDataset(
        input_ids,
        seq_len=seq_len,
        shift_labels=False,
        random_mask_offset=random_mask_offset,
        random_mask_prob=1.0,
        ignore_token_id=-100,
    )

    assert dataset[0]["input_ids"][1] == input_ids[1] - random_mask_offset
    assert dataset[1]["input_ids"][0] == input_ids[2] - random_mask_offset
    assert dataset[0]["labels"][1] == -100
    assert dataset[1]["labels"][0] == -100
