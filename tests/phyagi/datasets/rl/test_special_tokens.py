# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from phyagi.datasets.rl.special_tokens import get_mask_token, get_special_token


def test_get_special_token():
    assert get_special_token("phi", "assistant") == "<|assistant|>"
    assert get_special_token("chatml", "user") == "<|im_start|>user<|im_sep|>"

    with pytest.raises(ValueError):
        get_special_token("invalid_format")
    with pytest.raises(ValueError):
        get_special_token("phi", "invalid_type")


def test_get_mask_token():
    assert get_mask_token("start") == "<|mask_start|>"
    assert get_mask_token("end") == "<|mask_end|>"

    with pytest.raises(ValueError):
        get_mask_token("invalid_type")
