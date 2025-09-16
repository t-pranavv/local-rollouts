# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import datetime

from phyagi.utils.import_utils import is_vllm_available

if is_vllm_available():
    from vllm import LLM, SamplingParams
else:
    raise ImportError("vLLM is not available. Please install `vllm`.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates text with vLLM.")

    parser.add_argument(
        "pretrained_model_name_or_path",
        type=str,
        help="Pre-trained model name or path.",
    )

    parser.add_argument(
        "-t",
        "--pretrained_tokenizer_name_or_path",
        type=str,
        default="microsoft/phi-2",
        help="Pre-trained tokenizer name or path.",
    )

    parser.add_argument(
        "-tps", "--tensor-parallel-size", type=int, default=1, help="Number of GPUs to use for tensor parallelism."
    )

    parser.add_argument("-temp", "--temperature", type=float, default=0.8, help="Temperature to use for sampling.")

    parser.add_argument("-s", "--seed", type=int, default=None, help="Seed to use for sampling (time by default).")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_args()

    seed = datetime.datetime.now().microsecond if args.seed is None else args.seed
    llm = LLM(
        model=args.pretrained_model_name_or_path,
        tokenizer=args.pretrained_tokenizer_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=seed,
    )

    prompt = [
        '\n\ndef truncate_number(number: float) -> float:\n    """ Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    """\n',
        'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
        'from typing import List\n\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    """ For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case):\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    """\n',
        'from typing import List\n\n\ndef intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    """ Insert a number \'delimeter\' between every two consecutive elements of input list `numbers\'\n    >>> intersperse([], 4)\n    []\n    >>> intersperse([1, 2, 3], 4)\n    [1, 4, 2, 4, 3]\n    """\n',
        "from typing import List\n\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\" Filter an input list of strings only for ones that contain given substring\n    >>> filter_by_substring([], 'a')\n    []\n    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n    ['abc', 'bacd', 'array']\n    \"\"\"\n",
        'from typing import List, Tuple\n\n\ndef sum_product(numbers: List[int]) -> Tuple[int, int]:\n    """ For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.\n    Empty sum should be equal to 0 and empty product should be equal to 1.\n    >>> sum_product([])\n    (0, 1)\n    >>> sum_product([1, 2, 3, 4])\n    (10, 24)\n    """\n',
        'from typing import List, Tuple\n\n\ndef rolling_max(numbers: List[int]) -> List[int]:\n    """ From a given list of integers, generate a list of rolling maximum element found until given moment\n    in the sequence.\n    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n    [1, 2, 3, 3, 3, 4, 4]\n    """\n',
        "from typing import List\n\n\ndef string_xor(a: str, b: str) -> str:\n    \"\"\" Input are two strings a and b consisting only of 1s and 0s.\n    Perform binary XOR on these inputs and return result also as a string.\n    >>> string_xor('010', '110')\n    '100'\n    \"\"\"\n",
        "from typing import List, Optional\n\n\ndef longest(strings: List[str]) -> Optional[str]:\n    \"\"\" Out of list of strings, return the longest one. Return the first one in case of multiple\n    strings of the same length. Return None in case the input list is empty.\n    >>> longest([])\n\n    >>> longest(['a', 'b', 'c'])\n    'a'\n    >>> longest(['a', 'bb', 'ccc'])\n    'ccc'\n    \"\"\"\n",
        '\n\ndef greatest_common_divisor(a: int, b: int) -> int:\n    """ Return a greatest common divisor of two integers a and b\n    >>> greatest_common_divisor(3, 5)\n    1\n    >>> greatest_common_divisor(25, 15)\n    5\n    """\n',
    ]

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=80)
    outputs = llm.generate(prompt, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text

        # Prompt part is underlined
        print(f"\033[4m{prompt}\033[0m{generated_text}")
