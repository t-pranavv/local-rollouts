# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
from collections import Counter
from typing import Dict, List, Optional

import sympy
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

from phyagi.rl.rewards.math_utils.math_normalize import (
    normalize_latex_string,
    normalize_math_answer,
)

_BAD_SUBSTRINGS = ["^{", "^("]
_BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
_TUPLE_CHARS = "()[]"
_CHAT_TAG = "<|im_start|>assistant<|im_sep|>"

_PRED_CFG = LatexExtractionConfig(
    normalization_config=NormalizationConfig(
        nits=False,
        malformed_operators=False,
        basic_latex=True,
        boxed=True,
        units=True,
    ),
    boxed_match_priority=0,
    try_extract_without_anchor=False,
)
_GOLD_CFG = LatexExtractionConfig()


def _remove_thousands_separators(expr: str) -> str:
    pattern = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        updated_expr = pattern.sub("\\1\\3\\4", expr)
        if updated_expr == expr:
            break
        expr = updated_expr
    return expr


def _str_is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _float_is_int(value: float) -> bool:
    try:
        return abs(value - round(value)) <= 1e-7
    except Exception:
        return False


def _convert_latex_to_plain_text(expr: str) -> str:
    expr = expr.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac").replace(r"\frac", r" \frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    replacements = {
        "√": "sqrt",
        "π": "pi",
        "∞": "inf",
        "∪": "U",
        "·": "*",
        "×": "*",
    }
    for k, v in replacements.items():
        expr = expr.replace(k, v)

    return expr.strip()


def _convert_mixed_number_to_sum(expr: str) -> str:
    return re.sub(r"([0-9]) +([0-9])", r"\1+\2", expr)


def _str_is_int(value: str) -> bool:
    try:
        value = _remove_thousands_separators(value)
        value = float(value)
        return abs(value - int(round(value))) <= 1e-7
    except:
        return False


def _convert_str_to_int(value: str) -> int:
    value = value.replace(",", "")
    return int(float(value))


def _normalize_math_expression(expr: Optional[str]) -> Optional[str]:
    if expr is None:
        return None

    match = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if match:
        expr = match.group("text")

    expr = (
        expr.replace(r"\%", "%")
        .replace(r"\$", "$")
        .replace("$", "")
        .replace("%", "")
        .replace(" or ", " , ")
        .replace(" and ", " , ")
        .replace("million", "*10^6")
        .replace("billion", "*10^9")
        .replace("trillion", "*10^12")
    )

    expr = re.sub(
        r"(degree|cm|centimeter|meter|mile|second|minute|hour|day|week|month|year|foot|feet|inch|yard)(es)?(s)? *(\^[0-9]+)?",
        "",
        expr,
    )
    expr = re.sub(r"\^ *\\circ", "", expr)

    if expr.startswith("{") and expr.endswith("}"):
        expr = expr[1:-1]

    expr = re.sub(r",\\! *", "", expr)

    if _str_is_float(expr) and _float_is_int(float(expr)):
        expr = str(int(round(float(expr))))

    if "\\" in expr:
        try:
            expr = _convert_latex_to_plain_text(expr)
        except Exception:
            pass

    expr = expr.replace("- ", "-")
    expr = _convert_mixed_number_to_sum(expr).replace(" ", "")
    expr = expr.replace("{", "").replace("}", "").lower()

    if _str_is_int(expr):
        expr = str(_convert_str_to_int(expr))

    return expr


def _count_variables_in_expression(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def _expression_is_safe_for_eval(expr: str) -> bool:
    if _count_variables_in_expression(expr) > 2:
        return False
    if any(sub in expr for sub in _BAD_SUBSTRINGS):
        return False
    if any(re.search(regex, expr) for regex in _BAD_REGEXES):
        return False
    return True


def _sympy_parse_expression(expr: str):
    expr_py = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        expr_py,
        transformations=sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,),
    )


def _expressions_are_symbolically_equal(gt_norm: str, pred_norm: str) -> bool:
    try:
        expr = f"({gt_norm})-({pred_norm})"
        if not _expression_is_safe_for_eval(expr):
            return False
        diff = _sympy_parse_expression(expr)
        return sympy.simplify(diff) == 0
    except Exception:
        return False


def _split_if_wrapped_tuple(expr: str) -> List[str]:
    expr = _remove_thousands_separators(expr)
    if not expr:
        return []
    if (
        len(expr) > 2
        and expr[0] in _TUPLE_CHARS
        and expr[-1] in _TUPLE_CHARS
        and all(c not in expr[1:-1] for c in _TUPLE_CHARS)
    ):
        return [e.strip() for e in expr[1:-1].split(",")]
    return [expr]


def _is_fraction_str(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _grade_answer(given: str, gold: str) -> bool:
    if given is None:
        return False

    given_norm_math = normalize_math_answer(given)
    gold_norm_math = normalize_math_answer(gold)

    if given_norm_math == gold_norm_math:
        return True

    given_norm = _normalize_math_expression(given)
    gold_norm = _normalize_math_expression(gold)

    if gold_norm is None:
        return False
    if given_norm == gold_norm:
        return True
    if not given_norm:
        return False

    given_elems = _split_if_wrapped_tuple(given_norm)
    gold_elems = _split_if_wrapped_tuple(gold_norm)

    if (len(gold_elems) > 1 and (gold_norm[0] != given_norm[0] or gold_norm[-1] != given_norm[-1])) or (
        len(gold_elems) != len(given_elems)
    ):
        return False

    for gold_elem, given_elem in zip(gold_elems, given_elems):
        if _is_fraction_str(gold_elem) and _is_fraction_str(given_elem):
            if gold_elem != given_elem:
                return False
        elif _str_is_int(gold_elem) != _str_is_int(given_elem):
            return False
        elif not _expressions_are_symbolically_equal(gold_elem, given_elem):
            return False

    return True


def verify_latex_answer(answer: str, gold: str, kind: str = "math") -> Dict[str, object]:
    """Verify the LaTeX answer against the gold standard.

    Args:
        answer: The answer string to verify.
        gold: The gold standard answer string.
        kind: The type of the answer, e.g., "math", "writing".

    Returns:
        Dictionary containing the type of answer, LaTeX agreement score, and extracted values.

    """

    if kind == "writing":
        return {"type": "writing"}

    if _CHAT_TAG in answer:
        answer = answer.split(_CHAT_TAG, 1)[-1]

    gold_parsed = parse(f"${gold}$", extraction_config=[_GOLD_CFG]) or parse(gold)
    if not gold_parsed:
        return {"type": "math", "LaTeXAgreementScore": 0.0, "Extracted": ("N/A", "N/A")}

    ans_parsed = parse(answer, extraction_config=[_PRED_CFG]) or parse(answer)
    if not ans_parsed:
        return {"type": "math", "LaTeXAgreementScore": 0.0, "Extracted": ("N/A", str(gold_parsed[0]))}

    try:
        score = float(verify(ans_parsed[0], gold_parsed[0]))
    except Exception:
        score = 0.0

    return {"type": "math", "LaTeXAgreementScore": score, "Extracted": (answer, gold)}


def compute_repetition_penalty(text: str, n_gram_size: int = 5) -> float:
    """Compute the repetition penalty for a given text based on n-grams.

    Args:
        text: Input text to analyze.
        n_gram_size: Size of the n-grams to consider.

    Returns:
        A negative float value representing the repetition penalty, where higher absolute values
        indicate more repetition.

    """

    words = re.findall(r"\w+", text.lower())
    if len(words) < n_gram_size:
        return 0.0

    n_grams = list(zip(*[words[i:] for i in range(n_gram_size)]))
    n_gram_counts = Counter(n_grams)
    total_n_grams = len(n_grams)

    repeated_count = sum(1 for count in n_gram_counts.values() if count > 5)
    repeated_ratio = repeated_count / total_n_grams if total_n_grams > 0 else 0.0

    max_repetition = max(n_gram_counts.values())
    max_ratio = (max_repetition / (len(words) / n_gram_size)) if max_repetition >= 5 else 0.0

    return -max(repeated_ratio, max_ratio)


def are_answers_equivalent(str1: Optional[str], str2: Optional[str], verbose: bool = False) -> bool:
    """Check if two answers are equivalent, ignoring case and LaTeX formatting.

    Args:
        str1: First answer string.
        str2: Second answer string.
        verbose: Whether to print debug information.

    Returns:
        Whether the two answers are equivalent.

    """

    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    if str1.lower() == str2.lower() or _grade_answer(str1, str2):
        return True

    try:
        ss1 = normalize_latex_string(str1)
        ss2 = normalize_latex_string(str2)

        replacements = {"yes": "__YES__", "true": "__YES__", "no": "__NO__", "false": "__NO__"}
        ss1 = replacements.get(ss1.lower(), ss1)
        ss2 = replacements.get(ss2.lower(), ss2)

        if verbose:
            print(ss1, ss2)

        return ss1 == ss2
    except Exception:
        return False


def remove_boxed_wrapper(s: str) -> str:
    """Remove the LaTeX boxed wrapper from a string if it exists.

    Args:
        s: Input string that may contain a boxed expression.

    Returns:
        String without the boxed wrapper if it exists, otherwise raises ``ValueError``.

    """

    if s.startswith(r"\boxed "):
        return s[len(r"\boxed ") :]
    if s.startswith(r"\boxed{") and s.endswith("}"):
        return s[len(r"\boxed{") : -1]
    raise ValueError("Input string is not a valid boxed expression")
