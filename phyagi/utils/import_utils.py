# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib

import torch


def _is_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def is_flash_attn_available() -> bool:
    """Returns ``True`` if ``flash_attn`` is available."""

    return _is_package_available("flash_attn")


def is_flash_attn_3_available() -> bool:
    """Returns ``True`` if ``flash_attn_interface`` is available."""

    return _is_package_available("flash_attn_interface")


def is_fused_dense_lib_available() -> bool:
    """Returns ``True`` if ``fused_dense_lib`` is available."""

    return _is_package_available("fused_dense_lib")


def is_hopper_gpu_available() -> bool:
    """Returns ``True`` if the current GPU is a Hopper GPU (compute capability 9.x)."""

    if not torch.cuda.is_available():
        return False

    return torch.cuda.get_device_capability()[0] == 9


def is_lm_eval_available() -> bool:
    """Returns ``True`` if ``lm_eval`` is available."""

    return _is_package_available("lm_eval")


def is_mpi_available() -> bool:
    """Returns ``True`` if ``mpi4py`` is available."""

    return _is_package_available("mpi4py")


def is_torch_gpu_available() -> bool:
    """Returns ``True`` if ``torch.cuda.is_available()`` is ``True``."""

    return torch.cuda.is_available()


def is_torchao_available() -> bool:
    """Returns ``True`` if ``torchao`` is available."""

    return _is_package_available("torchao")


def is_vllm_available() -> bool:
    """Returns ``True`` if ``vllm`` is available."""

    return _is_package_available("vllm")
