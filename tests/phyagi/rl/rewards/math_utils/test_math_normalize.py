# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from phyagi.rl.rewards.math_utils.math_normalize import (
    normalize_latex_string,
    normalize_math_answer,
)


def test_normalize_latex_string():
    assert normalize_latex_string(r"\left(5\right)") == "(5)"
    assert normalize_latex_string(r"^{\circ}") == ""
    assert normalize_latex_string(r"0.5") == r"\frac{1}{2}"
    assert normalize_latex_string(r"3/4") == r"\frac{3}{4}"
    assert normalize_latex_string(r"\sqrt3") == r"\sqrt{3}"
    assert normalize_latex_string(r"\frac12") == r"\frac{1}{2}"
    assert normalize_latex_string(r"q = 42") == "42"
    assert normalize_latex_string(r"{.5}") == "{0.5}"
    assert normalize_latex_string(r"\$100") == "100"
    assert normalize_latex_string(r"50\%") == "50"
    assert normalize_latex_string("") == ""


def test_normalize_math_answer():
    assert normalize_math_answer("") == ""
    assert normalize_math_answer(None) is None
    assert normalize_math_answer(r"\text{42}") == "42"
    assert normalize_math_answer(r"  \text{ answer } ") == "answer"
    assert normalize_math_answer(r"42") == "42"
    assert normalize_math_answer(r"0.5") == r"\frac{1}{2}"
    assert normalize_math_answer(r"\frac12") == r"\frac{1}{2}"
    assert normalize_math_answer(r"3/4") == r"\frac{3}{4}"
    assert normalize_math_answer(r"\sqrt3") == r"\sqrt{3}"
    assert normalize_math_answer(r"^{\circ}") == ""
    assert normalize_math_answer(r"q = 42") == "42"
