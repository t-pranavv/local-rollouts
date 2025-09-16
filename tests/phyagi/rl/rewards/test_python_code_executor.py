# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock, patch

import pytest
import requests

from phyagi.rl.rewards.python_code_executor import PythonCodeExecutorReward


@pytest.fixture
def mock_azure_secret():
    with patch("phyagi.rl.rewards.python_code_executor.SecretClient") as mock_secret_client:
        mock_client_instance = MagicMock()
        mock_client_instance.get_secret.return_value.value = "mocked-token"
        mock_secret_client.return_value = mock_client_instance
        yield


@pytest.fixture
def mock_requests_post():
    with patch("phyagi.rl.rewards.python_code_executor.requests.post") as mock_post:
        yield mock_post


def test_score_success(mock_azure_secret, mock_requests_post):
    mock_requests_post.return_value.status_code = 200
    mock_requests_post.return_value.json.return_value = {"status": "succeeded"}

    reward = PythonCodeExecutorReward(service_url="https://fake-url.com")

    solution = "```python\nx = 2\n```"
    ground_truth = "assert x == 2"

    score = reward.score(solution, ground_truth)
    assert score == 1.0


def test_score_failure_status_code(mock_azure_secret, mock_requests_post):
    mock_requests_post.return_value.status_code = 500

    reward = PythonCodeExecutorReward(service_url="https://fake-url.com")

    solution = "```python\nx = 2\n```"
    ground_truth = "assert x == 2"

    score = reward.score(solution, ground_truth)
    assert score == reward._incorrect_reward


def test_score_failure_exec_status(mock_azure_secret, mock_requests_post):
    mock_requests_post.return_value.status_code = 200
    mock_requests_post.return_value.json.return_value = {"status": "failed"}

    reward = PythonCodeExecutorReward(service_url="https://fake-url.com")

    solution = "```python\nx = 2\n```"
    ground_truth = "assert x == 3"

    score = reward.score(solution, ground_truth)
    assert score == reward._incorrect_reward


def test_score_timeout(mock_azure_secret):
    with patch("phyagi.rl.rewards.python_code_executor.requests.post", side_effect=requests.exceptions.Timeout):
        reward = PythonCodeExecutorReward(service_url="https://fake-url.com")
        solution = "```python\nx = 2\n```"
        ground_truth = "assert x == 2"

        score = reward.score(solution, ground_truth)
        assert score == reward._incorrect_reward
