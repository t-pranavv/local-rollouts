# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from phyagi.rl.rewards.phi4rp import Phi4RPReward


@pytest.fixture(scope="module")
def reward_model():
    return Phi4RPReward(tokenizer="bert-base-uncased", max_response_length=1024)


@pytest.mark.parametrize(
    ("solution, ground_truth, expected_range"),
    [
        pytest.param(
            "<|im_start|>assistant<|im_sep|><think>\nWe solve the equation...\n</think>\n<answer>\n\\boxed{42}\n</answer><|im_end|>",
            "test######42",
            (0.9, 1.0),
            id="correct_solution",
        ),
        pytest.param(
            "<|im_start|>assistant<|im_sep|><think>\nSome wrong math...\n</think>\n<answer>\n\\boxed{13}\n</answer><|im_end|>",
            "test######42",
            (0.0, 0.1),
            id="incorrect_solution",
        ),
        pytest.param(
            "<|im_start|>assistant<|im_sep|><think>\nSome math...\n</think>\n<answer>\n42\n</answer><|im_end|>",
            "test######42",
            (0.0, 0.1),
            id="no_boxed",
        ),
        pytest.param(
            "<|im_start|>assistant<|im_sep|>\\boxed{42}<|im_end|>",
            "test######42",
            (0.9, 1.0),
            id="no_think",
        ),
        pytest.param(
            "<|im_start|>assistant<|im_sep|><think>\nUsing <tool>calculator</tool> we find...\n</think>\n<answer>\n\\boxed{42}\n</answer><|im_end|>",
            "train######42",
            (0.9, 1.0),
            id="tool_usage",
        ),
        pytest.param(
            "<|im_start|>assistant<|im_sep|><think>\nUsing <tool>long reasoning</tool>...\n"
            + "step\n" * 300
            + "</think>\n<answer>\n\\boxed{42}\n</answer><|im_end|>",
            "train######42",
            (0.5, 0.6),
            id="long_reasoning",
        ),
        pytest.param(
            "<|im_start|>assistant<|im_sep|><think>\nUsing <tool>wrong logic</tool>...\n</think>\n<answer>\n\\boxed{13}\n</answer><|im_end|>",
            "train######42",
            (-0.3, -0.2),
            id="wrong_tool_usage",
        ),
        pytest.param(
            "<|im_start|>assistant<|im_sep|><think>\nMissing end\n<answer>\n\\boxed{42}\n",
            "train######42",
            (-0.6, -0.5),
            id="missing_end_tag",
        ),
    ],
)
def test_phi4rp_reward(reward_model, solution, ground_truth, expected_range):
    score = reward_model.score(solution, ground_truth)
    assert expected_range[0] <= score <= expected_range[1]
