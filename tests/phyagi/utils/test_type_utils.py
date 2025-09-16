# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from phyagi.utils.type_utils import (
    nullcontext_kwargs,
    rgetattr,
    rsetattr,
    to_torch_dtype,
    xor,
)


@pytest.fixture
def obj():
    class DummyInnerObject:
        def __init__(self):
            self.attr = "some inner value"

    class DummyObject:
        def __init__(self):
            self.attr1 = DummyInnerObject()
            self.attr1.attr2 = DummyInnerObject()
            self.attr3 = "some value"

    return DummyObject()


def test_to_torch_dtype():
    assert to_torch_dtype("float16") == torch.float16
    assert to_torch_dtype("float32") == torch.float32
    assert to_torch_dtype("float64") == torch.float64
    assert to_torch_dtype("bfloat16") == torch.bfloat16


def test_xor():
    assert xor(True, True) is False
    assert xor(True, False) is True
    assert xor(False, True) is True
    assert xor(False, False) is False


def test_nullcontext_kwargs():
    with nullcontext_kwargs(foo=42, bar="test") as ctx:
        assert ctx is None

    try:
        with nullcontext_kwargs():
            raise ValueError("Test error")
    except ValueError as e:
        assert str(e) == "Test error"


def test_rgetattr(obj):
    assert rgetattr(obj, "attr3") == "some value"
    assert rgetattr(obj, "attr1.attr") == "some inner value"
    assert rgetattr(obj, "attr1.attr2.attr") == "some inner value"


def test_rsetattr(obj):
    rsetattr(obj, "attr3", "new value")
    assert obj.attr3 == "new value"

    rsetattr(obj, "attr1.attr", "some value")
    assert obj.attr1.attr == "some value"

    rsetattr(obj, "attr1.attr2.attr", "some value")
    assert obj.attr1.attr2.attr == "some value"
