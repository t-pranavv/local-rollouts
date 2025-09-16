# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Optional, Tuple, Union

import ray

from phyagi.rl.rewards.reward import Reward
from phyagi.utils.import_utils import is_vllm_available

CompletionOutput = Any
if is_vllm_available():
    from vllm import CompletionOutput


class _RewardActor:
    def __init__(self, rewards: Dict[str, Reward], num_retries: int = 0) -> None:
        self._rewards = rewards
        self._num_retries = num_retries

    def score(
        self,
        completion: str,
        ground_truth: str,
        reward_names: Optional[List[str]] = None,
        reward_weights: Optional[List[float]] = None,
        key: Optional[Any] = None,
    ) -> Union[Dict[str, float], Tuple[Any, Dict[str, float]]]:
        reward_names = reward_names or list(self._rewards.keys())
        reward_weights = dict(zip(reward_names, reward_weights)) or {r: 1.0 / len(reward_names) for r in reward_names}

        scores = {}
        for reward_name in reward_names:
            for attempt in range(self._num_retries + 1):
                try:
                    scores[reward_name] = self._rewards[reward_name](completion, ground_truth)
                    break
                except Exception as e:
                    print(f"Error in reward {reward_name}: {e}. Attempt {attempt}/{1+self._num_retries}...")
                    scores[reward_name] = None

        scores["total_reward"] = (
            None
            if any(score is None for score in scores.values())
            else sum(score * reward_weights[reward_name] for reward_name, score in scores.items())
        )

        if key:
            return key, scores

        return scores


class RewardManager:
    """Manages multiple reward functions and distributes the workload across multiple workers."""

    def __init__(self, rewards: Dict[str, Reward], num_workers: int = 1) -> None:
        """Initialize the reward manager.

        Args:
            rewards: Dictionary of reward names and their corresponding reward functions.
            num_workers: Number of workers to use for parallel processing.

        """

        self._num_workers = num_workers
        self._rewards = rewards

        if num_workers > 1:
            worker_cls = ray.remote(num_cpus=1.0)(_RewardActor)
            self._actors = [worker_cls.remote(rewards) for _ in range(num_workers)]
        else:
            self._actors = [_RewardActor(rewards)]

    def score(
        self,
        generations: List[List[CompletionOutput]],  # type: ignore
        ground_truths: List[str],
        reward_names: List[List[str]],
        reward_weights: Optional[List[List[float]]] = None,
    ) -> List[List[Dict[str, float]]]:
        """Score the generated completions against the ground truth using the specified rewards.

        Args:
            generations: Generated completions.
            ground_truths: Ground truth strings.
            reward_names: Reward names to use for scoring.
            reward_weights: Weights for each reward.

        Returns:
            Dictionaries containing the scores for each reward.

        """

        if not all(r in self._rewards for reward_list in reward_names for r in reward_list):
            raise ValueError("All reward names must be in the rewards dictionary.")
        if len(generations) != len(ground_truths):
            raise ValueError(
                f"Number of generations must match number of ground truths, but got {len(generations)} and {len(ground_truths)})."
            )

        futures = []
        num_generations = 0

        # Round-robin the actors to distribute the workload
        for q_idx, (question_completions, ground_truth) in enumerate(zip(generations, ground_truths)):
            question_futures = []

            weights = reward_weights[q_idx] if reward_weights else None
            rewards = reward_names[q_idx]

            for c_idx, completion in enumerate(question_completions):
                actor = self._actors[num_generations % self._num_workers]

                question_futures.append(
                    actor.score.remote(completion.text, ground_truth, rewards, weights, key=(q_idx, c_idx))
                    if self._num_workers > 1
                    else actor.score(completion.text, ground_truth, rewards, weights)
                )

                num_generations += 1

            futures.append(question_futures)

        if self._num_workers <= 1:
            return futures

        # Flatten the futures list and get the results
        flat_futures = [completion_future for question_futures in futures for completion_future in question_futures]
        scores = ray.get(flat_futures)

        # Reshape the results back to the original structure
        results = [[None for _ in range(len(question_futures))] for question_futures in futures]
        for (q_idx, c_idx), score_dict in scores:
            results[q_idx][c_idx] = score_dict

        return results
