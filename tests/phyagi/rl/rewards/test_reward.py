# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock

from phyagi.rl.rewards.reward import Reward


class MyReward(Reward):
    def __init__(self) -> None:
        pass

    def score(self) -> float:
        return MagicMock()


def test_my_reward():
    reward = MyReward()

    score = reward.score()
    assert score is not None
