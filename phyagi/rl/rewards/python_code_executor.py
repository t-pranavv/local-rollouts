# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import requests
from azure.identity import AzureCliCredential, ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

from phyagi.rl.rewards.reward import Reward


class PythonCodeExecutorReward(Reward):
    """Python code executor reward."""

    def __init__(
        self,
        service_url: str,
        incorrect_reward: float = 0.0,
        auth_token: Optional[str] = None,
        managed_identity: Optional[str] = None,
        extract_markdown: bool = True,
        max_runtime: float = 10.0,
        max_request_timeout: float = 30.0,
    ) -> None:
        """Initialize the reward.

        Args:
            service_url: URL of the remote Python executor service.
            incorrect_reward: Reward for incorrect solutions.
            auth_token: Authentication token for the remote service.
            managed_identity: Managed identity for Azure authentication.
            extract_markdown: Whether to extract Python code from markdown.
            max_runtime: Maximum runtime for the code execution.
            max_request_timeout: Maximum timeout for the request.

        """

        self._service_url = service_url
        self._incorrect_reward = incorrect_reward
        self._extract_markdown = extract_markdown
        self._max_runtime = max_runtime
        self._max_request_timeout = max_request_timeout

        if auth_token is None:
            credential = (
                AzureCliCredential()
                if managed_identity is None
                else ManagedIdentityCredential(client_id=managed_identity)
            )

            kv = SecretClient(vault_url="https://phi-exec-kv.vault.azure.net/", credential=credential)
            auth_token = kv.get_secret("CodeExecApiKey").value

        self._token = auth_token

    def score(self, solution: str, ground_truth: str) -> float:
        if self._extract_markdown:
            splits = solution.split("```python")
            if len(splits) > 1:
                solution = splits[1].split("```")[0].strip()

        try:
            solution_response = requests.post(
                self._service_url,
                headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
                json={
                    "code": f"{solution}\n{ground_truth}",
                    "max_runtime": self._max_runtime,
                },
                timeout=self._max_request_timeout,
            )
        except requests.exceptions.Timeout:
            print(f"Code execution request timed out. Returning {self._incorrect_reward=}.")
            return self._incorrect_reward

        if solution_response.status_code != 200:
            print(f"Code execution request failed with status code {solution_response.status_code}.")
            return self._incorrect_reward

        solution_response = solution_response.json()
        if solution_response["status"] != "succeeded":
            return self._incorrect_reward

        return 1.0
