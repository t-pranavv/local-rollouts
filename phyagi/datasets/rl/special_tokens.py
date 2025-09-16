# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Dict, Optional, Union

SPECIAL_TOKENS = {
    "phi": {
        "assistant": "<|assistant|>",
        "user": "<|user|>",
        "end": "<|end|>",
        "system": "<|system|>",
    },
    "chatml": {
        "assistant": "<|im_start|>assistant<|im_sep|>",
        "user": "<|im_start|>user<|im_sep|>",
        "end": "<|im_end|>",
        "system": "<|im_start|>system<|im_sep|>",
    },
}

MASK_TOKENS = {
    "start": "<|mask_start|>",
    "end": "<|mask_end|>",
}


def get_special_token(
    special_token_format: str, special_token_type: Optional[str] = None
) -> Union[str, Dict[str, str]]:
    """Get a special token or a dictionary of special tokens.

    Args:
        special_token_format: Format of the special tokens.
        special_token_type: Type of the special token.
            If ``None``, returns all special tokens for the given format.
            If specified, must be one of the keys in the dictionary for the given format.

    Returns:
        Special token or a dictionary of special tokens.

    """

    if special_token_format not in SPECIAL_TOKENS:
        raise ValueError(
            f"`special_token_format` must be one of {list(SPECIAL_TOKENS.keys())}, but got '{special_token_format}'."
        )

    special_tokens = SPECIAL_TOKENS[special_token_format]
    if special_token_type is None:
        return special_tokens

    if special_token_type not in special_tokens:
        raise ValueError(
            f"`special_token_type` must be one of {list(special_tokens.keys())}, but got '{special_token_type}'."
        )

    return special_tokens[special_token_type]


def get_mask_token(mask_token_type: str) -> str:
    """Get a mask token.

    Args:
        mask_token_type: Type of the mask token.

    Returns:
        Mask token.

    """

    if mask_token_type not in MASK_TOKENS:
        raise ValueError(f"`mask_token_type` must be one of {list(MASK_TOKENS.keys())}, but got '{mask_token_type}'.")
    return MASK_TOKENS[mask_token_type]
