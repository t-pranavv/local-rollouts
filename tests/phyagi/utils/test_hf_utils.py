# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch.nn as nn

from phyagi.utils.hf_utils import to_device_map


@pytest.fixture
def model():
    return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))


@pytest.mark.parametrize("device_map", ["cpu"])
def test_to_device_map_without_gpu(model, device_map):
    original_state_dict = model.state_dict()

    transformed_model = to_device_map(model, device_map=device_map)
    transformed_state_dict = transformed_model.state_dict()

    assert set(original_state_dict.keys()) == set(transformed_state_dict.keys())
    assert set(v.device for v in original_state_dict.values()) == set(v.device for v in transformed_state_dict.values())


@pytest.mark.is_torch_gpu
@pytest.mark.parametrize("device_map", ["auto", "balanced", "balanced_low_0", "cuda", "sequential"])
def test_to_device_map_with_gpu(model, device_map):
    original_state_dict = model.state_dict()

    transformed_model = to_device_map(model, device_map=device_map)
    transformed_state_dict = transformed_model.state_dict()

    assert set(original_state_dict.keys()) == set(transformed_state_dict.keys())
    assert set(v.device for v in original_state_dict.values()) != set(v.device for v in transformed_state_dict.values())
