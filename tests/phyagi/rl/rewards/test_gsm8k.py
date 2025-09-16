# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from phyagi.rl.rewards.gsm8k import GSM8kReward


@pytest.mark.parametrize(
    "method,solution,ground_truth,expected_score",
    [
        ("strict", "The answer is #### 42", "42", 1.0),
        ("strict", "We compute it step by step. #### 41", "42", 0.0),
        ("strict", "No final answer here.", "42", 0.0),
        ("flexible", "We tried several paths. Eventually we got 10. Then 42.", "42", 1.0),
        ("flexible", "We tried 5, then concluded with 40.", "42", 0.0),
        ("flexible", "Just words and symbols!", "42", 0.0),
    ],
)
def test_gsm8k_reward_scoring(method, solution, ground_truth, expected_score):
    reward = GSM8kReward(method=method, format_score=0.0, correct_score=1.0)

    score = reward.score(solution, ground_truth)
    assert score == expected_score
