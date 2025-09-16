# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import functools
from contextlib import AbstractContextManager
from typing import Any, Union

import torch


def to_torch_dtype(dtype: str) -> torch.dtype:
    """Convert a string dtype to a PyTorch dtype.

    Args:
        dtype: String representation of dtype.

    Returns:
        PyTorch representation of dtype.

    """

    if dtype in ["torch.float16", "float16", "half", "fp16"]:
        return torch.float16

    if dtype in ["torch.float32", "float32", "float", "fp32"]:
        return torch.float32

    if dtype in ["torch.float64", "float64", "double", "fp64"]:
        return torch.float64

    if dtype in ["torch.bfloat16", "bfloat16", "bf16"]:
        return torch.bfloat16

    return torch.float32


def xor(p: Union[str, bool], q: Union[str, bool]) -> bool:
    """Logical XOR operator.

    Args:
        p: First operand.
        q: Second operand.

    Returns:
        Result of the XOR operation.

    """

    return (p and not q) or (not p and q)


def nullcontext_kwargs(**kwargs) -> AbstractContextManager:
    """Context manager that does nothing but accepts keyword arguments."""

    class _nullcontext(AbstractContextManager):
        def __enter__(self, *args, **kwargs) -> None:
            pass

        def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
            pass

    return _nullcontext()


def rgetattr(obj: Any, attr: str, *args) -> Any:
    """Recursively get an attribute from an object.

    This function allows accessing nested attributes by separating each level with a dot
    (e.g., "attr1.attr2.attr3"). If any attribute along the chain does not exist, the function
    returns the default value specified in the ``*args`` parameter.

    Examples:
        >>> obj = MyObject()
        >>> rgetattr(obj, "attr1.attr2.attr3")

    Args:
        obj: Object from which the attribute will be retrieved.
        attr: Name of the attribute to be retrieved, with each level separated by a dot.

    Returns:
        Attribute from the object.

    """

    def _getattr(obj: Any, attr: Any) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj: Any, attr: str, value: Any) -> None:
    """Recursively set an attribute on an object.

    This function allows setting nested attributes by separating each level with a dot (e.g., "attr1.attr2.attr3").

    Examples:
        >>> obj = MyObject()
        >>> rsetattr(obj, "attr1.attr2.attr3", new_value)

    Args:
        obj: Object on which the attribute will be set.
        attr: Name of the attribute to be set, with each level separated by a dot.
        value: New value for the attribute.

    """

    pre_attr, _, post_attr = attr.rpartition(".")

    return setattr(rgetattr(obj, pre_attr) if pre_attr else obj, post_attr, value)
