# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


class Reward:
    """Abstract class for rewards.

    This class serves as a base for implementing rewards that can be used to
    evaluate the performance of a solution against a ground truth. It enforces
    implementation of the :meth:`score` method, which takes a solution and
    ground truth as input and returns a score.

    Examples:
        >>> class MyReward(Reward):
        >>>     def __init__(self) -> None:
        >>>         pass
        >>>
        >>>     def score(self, solution: str, ground_truth: str) -> float:
        >>>         return 1.0 if solution == ground_truth else 0.0

    """

    def __init__(self) -> None:
        """Initialize the reward."""

        pass

    def __call__(self, solution: str, ground_truth: str) -> float:
        return self.score(solution, ground_truth)

    def score(self, solution: str, ground_truth: str) -> float:
        """Calculate the score of a solution against the ground truth.

        Args:
            solution: Solution to be evaluated.
            ground_truth: Ground truth to compare against.

        Returns:
            Score of the solution against the ground truth.

        """

        raise NotImplementedError("`Reward` must implement `score()`.")
