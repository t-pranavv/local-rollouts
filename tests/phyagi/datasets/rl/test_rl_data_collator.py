# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from phyagi.datasets.rl.rl_data_collator import RewardDataCollator


def test_reward_data_collator():
    collator = RewardDataCollator(reward_names=["reward1", "reward2"], reward_weights=[0.4, 0.6])

    x = torch.tensor([1, 2, 3, 10, 16, 20, 21])
    examples = [{"input_ids": x}, {"input_ids": x}]
    examples = collator(examples)

    assert examples["reward_names"][0] == ["reward1", "reward2"]
    assert examples["reward_weights"][0] == [0.4, 0.6]
    assert torch.equal(examples["input_ids"][0], x)


def test_reward_data_collator_reward_weights_calculation():
    collator = RewardDataCollator(reward_names=["r1", "r2", "r3"])

    x = torch.tensor([1, 2])
    result = collator([{"input_ids": x}])

    assert result["reward_weights"][0] == [1 / 3, 1 / 3, 1 / 3]


def test_reward_data_collator_invalid_arguments():
    with pytest.raises(ValueError):
        RewardDataCollator(reward_names=["r1", "r2"], reward_weights=[0.5])
    with pytest.raises(ValueError):
        RewardDataCollator(reward_names=[])
