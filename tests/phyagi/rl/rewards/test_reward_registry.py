# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from phyagi.rl.rewards.gsm8k import GSM8kReward
from phyagi.rl.rewards.registry import get_reward
from phyagi.rl.rewards.reward import Reward


def test_get_reward():
    reward_config = {"name": "test", "type": "gsm8k"}

    reward = get_reward(reward_config)
    assert isinstance(reward["test"], (Reward, GSM8kReward))
