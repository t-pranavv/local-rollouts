# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from types import SimpleNamespace

import pytest
import ray

from phyagi.rl.rewards.gsm8k import GSM8kReward
from phyagi.rl.rewards.reward_manager import RewardManager, _RewardActor


@pytest.fixture(scope="module", autouse=True)
def ray_init_shutdown():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_reward_actor_single_score():
    rewards = {"gsm8k": GSM8kReward()}
    actor = _RewardActor(rewards)
    result = actor.score("answer", "answer", reward_names=["gsm8k"], reward_weights=[1.0])

    assert isinstance(result, dict)
    assert result["gsm8k"] == 0.0
    assert result["total_reward"] == 0.0


def test_reward_actor_weighted_score():
    rewards = {
        "r1": GSM8kReward(),
        "r2": lambda c, g: 1.0,
    }
    actor = _RewardActor(rewards)
    result = actor.score("answer", "answer", reward_names=["r1", "r2"], reward_weights=[0.7, 0.3])

    assert result["r1"] == 0.0
    assert result["r2"] == 1.0
    assert result["total_reward"] == pytest.approx(0.3)


def test_reward_manager_single_worker():
    rewards = {"gsm8k": GSM8kReward()}
    manager = RewardManager(rewards, num_workers=1)

    generations = [[SimpleNamespace(text="hello"), SimpleNamespace(text="hi")]]
    ground_truths = ["hello"]
    reward_names = [["gsm8k"]]
    reward_weights = [[1.0]]

    results = manager.score(generations, ground_truths, reward_names, reward_weights)

    assert len(results) == 1
    assert len(results[0]) == 2
    assert results[0][0]["gsm8k"] == 0.0
    assert results[0][1]["gsm8k"] == 0.0
    assert results[0][0]["total_reward"] == 0.0
    assert results[0][1]["total_reward"] == 0.0


def test_reward_manager_multiple_workers():
    rewards = {"gsm8k": GSM8kReward()}
    manager = RewardManager(rewards, num_workers=2)

    generations = [[SimpleNamespace(text="correct"), SimpleNamespace(text="wrong")], [SimpleNamespace(text="correct")]]
    ground_truths = ["correct", "correct"]
    reward_names = [["gsm8k"], ["gsm8k"]]
    reward_weights = [[1.0], [1.0]]

    results = manager.score(generations, ground_truths, reward_names, reward_weights)

    assert len(results) == 2
    assert results[0][0]["total_reward"] == 0.0
    assert results[0][1]["total_reward"] == 0.0
    assert results[1][0]["total_reward"] == 0.0
