# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from phyagi.rl.rewards.math_utils.math_verify import (
    are_answers_equivalent,
    compute_repetition_penalty,
    remove_boxed_wrapper,
    verify_latex_answer,
)


def test_are_answers_equivalent():
    assert are_answers_equivalent("YES", "True") is True
    assert are_answers_equivalent("NO", "False") is True
    assert are_answers_equivalent("5", "5") is True
    assert are_answers_equivalent("5", "6") is False


def test_compute_repetition_penalty():
    text = "the cat sat on the mat the cat sat on the mat the cat sat on the mat"
    penalty = compute_repetition_penalty(text, n_gram_size=3)
    assert penalty <= 0

    text_no_repetition = "this is a unique sentence with no repetition at all"
    penalty = compute_repetition_penalty(text_no_repetition, n_gram_size=3)
    assert penalty == 0


def test_remove_boxed_wrapper():
    assert remove_boxed_wrapper(r"\boxed 42") == "42"
    assert remove_boxed_wrapper(r"\boxed{42}") == "42"


def test_verify_latex_answer():
    result = verify_latex_answer("4", "2+2")
    assert isinstance(result, dict)
    assert result["type"] == "math"
    assert result["LaTeXAgreementScore"] >= 0.0
