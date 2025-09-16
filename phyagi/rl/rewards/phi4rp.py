# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import re
from typing import List, Optional, Union

from transformers import AutoTokenizer, PreTrainedTokenizer

from phyagi.rl.rewards.math_utils.math_verify import (
    are_answers_equivalent,
    compute_repetition_penalty,
    remove_boxed_wrapper,
    verify_latex_answer,
)
from phyagi.rl.rewards.reward import Reward


def _count_xml(text: str) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.25
    if text.count("\n</think>\n") == 1:
        count += 0.25
    if text.count("\n<answer>\n") == 1:
        count += 0.25
    if text.count("\n</answer>") == 1:
        count += 0.25
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def _soft_format_reward_fn(completions: List[str]) -> List[float]:
    pattern = r"<think>.*?</think>.*<answer>.*</answer>.*<\|im_end\|>"
    responses = [completion for completion in completions]

    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [1.0 if match else 0.0 for match in matches]


def _last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")

    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]

    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    return string[idx : right_brace_idx + 1]


def _compute_score(solution_str: str, ground_truth: str) -> float:
    # Use `re` to extract the answer from the solution string
    # (<answer>...</answer> tags and multiple lines)
    pattern = r"<answer>.*?</answer>"
    matches = re.findall(pattern, solution_str, re.DOTALL)

    extracted_sol = None
    if len(matches) == 1:
        extracted_sol = matches[0]
    else:
        string_in_last_boxed = _last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            try:
                extracted_sol = string_in_last_boxed
                if extracted_sol.strip() == "\\boxed{}":
                    return 0.0

                # Checks whether completion is correct
                answer = remove_boxed_wrapper(string_in_last_boxed)
                if are_answers_equivalent(answer, ground_truth):
                    return 1.0
                elif answer.isnumeric() and ground_truth.isnumeric():
                    return 0.0

            except Exception:
                pass

    if extracted_sol is None:
        return 0.0

    response = verify_latex_answer(extracted_sol, ground_truth)

    return response["LaTeXAgreementScore"]


class Phi4RPReward(Reward):
    """Phi-4-reasoning-plus reward."""

    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer],
        max_response_length: int = 12976,
        max_length_control: Optional[int] = None,
        unit_length: Optional[int] = None,
    ) -> None:
        """Initialize the reward.

        Args:
            tokenizer: Tokenizer to use for encoding the responses.
            max_response_length: Maximum length of the response.
            max_length_control: Maximum length for controlling the reward scaling.
            unit_length: Unit length for scaling the reward.
                If not provided, it will be set to a value based on ``max_response_length``.

        """

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer) if isinstance(tokenizer, str) else tokenizer
        self._max_response_length = max_response_length
        self._unit_length = unit_length or max(max_response_length // 4 * 3, max_response_length - 3702)
        self._max_length_control = max_length_control or self._unit_length

    def score(self, solution: str, ground_truth: str) -> float:
        if len(ground_truth.split("######")) != 2:
            raise ValueError(
                f"`ground_truth` must be in the format 'solution######ground_truth', but got '{ground_truth}'."
            )

        response_length = len(self._tokenizer(solution).input_ids)
        split, ground_truth = ground_truth.split("######")

        # Remove prompt from completion
        if "<|im_start|>assistant<|im_sep|>" in solution:
            solution = solution.split("<|im_start|>assistant<|im_sep|>")[1]

        soft_format_reward = _soft_format_reward_fn([solution])[0]
        xml_reward = _count_xml(solution)

        # Remove end-of-text tag from completion
        solution = solution.replace("<|im_end|>", "").strip()

        # If we are in the test split, we directly compare the solution with the ground truth
        if split == "test":
            solution = _last_boxed_only_string(solution)
            if solution is not None:
                return _compute_score(solution, ground_truth)
            return 0.0

        # If there are more than one think tags, or if the think tag is not present,
        # we return -1 to encourage the model to use the think tag and prevent reward hacking
        invalid_think = solution.count("<think>") > 1 or solution.count("</think>") > 1
        no_think = "<think>" not in solution or "</think>" not in solution

        progress = 0.0
        weights = [2.0, 0.25, 0.5, 0.25, 0.25]
        repetition_penalty_score = compute_repetition_penalty(solution)
        tool_reward = 0.0

        if "<think>" in solution:
            inside_think_tag = solution.split("<think>")[-1].split("</think>")[0]
            if "<tool>" in inside_think_tag and "</tool>" in inside_think_tag:
                tool_reward = 1.0

        if invalid_think or no_think:
            # If the model does not use the think tag, we don't want to reward it for using a tool
            rewards = [-1.0, repetition_penalty_score, soft_format_reward, xml_reward, 0.0]
            return sum([r * w for r, w in zip(rewards, weights)]) / sum(weights)

        # If the completion does not contain the think tag, we try to encourage the model to use
        # the think tag and prevent reward hacking
        if "</think>" not in solution:
            reward = -1.0
        else:
            min_value_wrong, max_value_wrong = -1.0, -0.5
            min_value_correct, max_value_correct = 0.5, 1.0

            # Remove the think tag from the completion only for training
            solution = solution.split("</think>")[-1]

            score = _compute_score(solution, ground_truth)
            if score > 0.5:
                min_value, max_value = min_value_correct, max_value_correct
                progress = min(
                    1,
                    max(response_length - self._max_length_control, 0)
                    / (self._max_response_length - self._max_length_control),
                )
            else:
                # Swap min/max for incorrect answers
                min_value, max_value = max_value_wrong, min_value_wrong
                progress = min(1, response_length / (self._max_response_length - self._unit_length))

            cosine = math.cos(progress * math.pi)
            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)

        return sum(
            [
                r * w
                for r, w in zip(
                    [reward, repetition_penalty_score, soft_format_reward, xml_reward, tool_reward], weights
                )
            ]
        ) / sum(weights)
