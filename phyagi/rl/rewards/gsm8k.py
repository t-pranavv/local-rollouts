# Copyright 2024 Bytedance Ltd. and/or its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License").

import re

from math_verify import ExprExtractionConfig, parse, verify

from phyagi.rl.rewards.reward import Reward


def _extract_solution(solution: str, method: str = "strict") -> str:
    if method == "strict":
        answer = re.search("#### ?(\\-?[0-9\\.\\,]+)", solution)
        if answer is None:
            output_answer = None
        else:
            output_answer = answer.group(0).split("####")[1].replace(",", "").replace("$", "").strip()

    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution)
        output_answer = None
        if len(answer) == 0:
            pass
        else:
            invalid_str = ["", "."]
            for output_answer in reversed(answer):
                if output_answer.strip() not in invalid_str:
                    break

    return output_answer


def _compute_score(
    solution: str, ground_truth: str, method: str = "strict", format_score: float = 0.0, score: float = 1.0
) -> float:
    answer = _extract_solution(solution=solution, method=method)
    if answer is None:
        return 0

    answer = parse(answer, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(ground_truth, extraction_config=[ExprExtractionConfig()])

    correct_answer = verify(ground_truth, answer)
    if correct_answer:
        return score

    return format_score


class GSM8kReward(Reward):
    """Reward function for GSM8k-style arithmetic problems.

    This reward function extracts the answer from the model-generated solution and compares
    it to the ground truth answer. It supports two methods of extraction:
     - "strict": Requires an exact match of the answer format.
     - "flexible": Allows for some flexibility in the answer format.

    The reward is computed based on whether the extracted answer matches the ground truth.
    If the answer is correct, it returns a score of `correct_score`, otherwise it returns
    `format_score`.

    """

    def __init__(self, method: str = "strict", format_score: float = 0.0, correct_score: float = 1.0) -> None:
        """Initialize the reward.

        Args:
            method: Method to use for extracting the answer from the solution.
            format_score: Score to return if the answer is not correct.
            correct_score: Score to return if the answer is correct.

        """

        if method not in ["strict", "flexible"]:
            raise ValueError(f"`method` should be either 'strict' or 'flexible', but got '{method}'.")

        self._method = method
        self._format_score = format_score
        self._correct_score = correct_score

    def score(self, solution: str, ground_truth: str) -> float:
        return _compute_score(
            solution=solution,
            ground_truth=ground_truth,
            method=self._method,
            format_score=self._format_score,
            score=self._correct_score,
        )
