# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from phyagi.datasets.train.train_data_collator import LMDataCollator


@pytest.fixture
def sample_tensor():
    return torch.tensor([1, 2, 3, 10, 16, 20, 21])


def test_lm_data_collator(sample_tensor):
    collator = LMDataCollator(ignore_token_id_range=(1, 10), ignore_token_ids=[15, 20], ignore_index=-200)
    examples = [{"input_ids": sample_tensor, "labels": sample_tensor.clone()} for _ in range(2)]
    batch = collator(examples)

    expected_labels = torch.tensor([-200, -200, -200, -200, 16, -200, 21])
    for idx in range(2):
        assert torch.equal(batch["input_ids"][idx], sample_tensor)
    for idx in range(2):
        assert torch.equal(batch["labels"][idx], expected_labels)


def test_lm_data_collator_without_ignore_attributes(sample_tensor):
    collator = LMDataCollator(ignore_token_ids=None, ignore_token_id_range=None, ignore_index=-200)
    examples = [{"input_ids": sample_tensor, "labels": sample_tensor.clone()}]
    batch = collator(examples)

    assert torch.equal(batch["labels"][0], sample_tensor)
