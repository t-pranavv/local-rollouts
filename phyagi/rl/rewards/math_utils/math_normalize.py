# Copyright (c) hendrycks/math.
# Licensed under the MIT license.

import re
from typing import Optional


def _wrap_unbraced_frac_arguments(string: str) -> str:
    parts = string.split("\\frac")
    fixed_string = parts[0]

    for part in parts[1:]:
        fixed_string += "\\frac"

        if part.startswith("{"):
            fixed_string += part
            continue

        if len(part) < 2:
            return string

        numerator = part[0]
        denominator = part[1]

        if denominator != "{":
            rest = part[2:] if len(part) > 2 else ""
            fixed_string += f"{{{numerator}}}{{{denominator}}}{rest}"
        else:
            rest = part[2:] if len(part) > 2 else ""
            fixed_string += f"{{{numerator}}}{denominator}{rest}"

    return fixed_string


def _convert_simple_fraction_to_latex(string: str) -> str:
    components = string.split("/")
    if len(components) != 2:
        return string

    numerator, denominator = components
    try:
        numerator_int = int(numerator)
        denominator_int = int(denominator)
        assert string == f"{numerator_int}/{denominator_int}"
        return f"\\frac{{{numerator_int}}}{{{denominator_int}}}"
    except Exception:
        return string


def _remove_trailing_text_units(string: str) -> str:
    if "\\text{ " in string:
        parts = string.split("\\text{ ")
        if len(parts) == 2:
            return parts[0]
    return string


def _wrap_unbraced_sqrt_arguments(string: str) -> str:
    if "\\sqrt" not in string:
        return string

    parts = string.split("\\sqrt")
    fixed_string = parts[0]

    for part in parts[1:]:
        if part and not part.startswith("{"):
            fixed_string += f"\\sqrt{{{part[0]}}}{part[1:]}"
        else:
            fixed_string += f"\\sqrt{part}"

    return fixed_string


def normalize_latex_string(string: str) -> str:
    """Normalize a LaTeX string by removing unnecessary formatting and simplifying the expression.

    Args:
        string: LaTeX string to normalize.

    Returns:
        Normalized string representation of the LaTeX expression.

    """

    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_trailing_text_units(string)
    string = string.replace("\\%", "").replace("\%", "")

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if string.startswith("."):
        string = "0" + string

    if "=" in string:
        left, right = string.split("=", maxsplit=1)
        if len(left.strip()) <= 2:
            string = right.strip()

    string = _wrap_unbraced_sqrt_arguments(string)
    string = string.replace(" ", "")
    string = _wrap_unbraced_frac_arguments(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = _convert_simple_fraction_to_latex(string)

    return string


def normalize_math_answer(answer: Optional[str]) -> Optional[str]:
    """Normalize a math answer by stripping LaTeX formatting and simplifying the expression.

    Args:
        answer: Math answer to normalize, which may be in LaTeX format.

    Returns:
        Normalized string representation of the math answer or ``None`` if the input is ``None``.

    """

    if answer is None:
        return None

    answer = answer.strip()
    try:
        match = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if match:
            answer = match.group("text").strip()
        return normalize_latex_string(answer)
    except Exception:
        return answer
