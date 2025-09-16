# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Any, Dict, List, Union

from phyagi.rl.rewards.gsm8k import GSM8kReward
from phyagi.rl.rewards.phi4rp import Phi4RPReward
from phyagi.rl.rewards.python_code_executor import PythonCodeExecutorReward
from phyagi.rl.rewards.reward import Reward
from phyagi.utils.logging_utils import get_logger

logger = get_logger(__name__)

REWARDS = {
    "gsm8k": GSM8kReward,
    "python": PythonCodeExecutorReward,
    "phi4rp": Phi4RPReward,
}


def get_reward(reward_configs: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Reward]:
    """Get rewards.

    Args:
        reward_configs: Reward configuration.

    Returns:
        Dictionary of rewards.

    """

    logger.info("Loading rewards...")
    reward_configs = [reward_configs] if not isinstance(reward_configs, list) else reward_configs
    logger.info(f"Rewards: {reward_configs}")

    rewards = {}
    for i, rc in enumerate(reward_configs):
        reward_type = rc.get("type", None)
        if reward_type not in REWARDS:
            raise ValueError(f"`type` must be one of {list(REWARDS.keys())}, but got '{reward_type}'.")

        reward_kwargs = rc.get("kwargs", {})
        reward = REWARDS[reward_type](**reward_kwargs)

        reward_name = rc.get("name", f"reward_{i}")
        rewards[reward_name] = reward

    return rewards
