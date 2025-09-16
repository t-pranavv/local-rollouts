# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch

import pytest

from phyagi.utils.import_utils import (
    is_flash_attn_3_available,
    is_flash_attn_available,
    is_fused_dense_lib_available,
    is_hopper_gpu_available,
    is_lm_eval_available,
    is_mpi_available,
    is_torch_gpu_available,
    is_torchao_available,
    is_vllm_available,
)


@pytest.mark.parametrize(
    "fn",
    [
        (is_flash_attn_available),
        (is_flash_attn_3_available),
        (is_fused_dense_lib_available),
        (is_lm_eval_available),
        (is_mpi_available),
        (is_torchao_available),
        (is_vllm_available),
    ],
)
def test_is_package_available(fn):
    with patch("importlib.util.find_spec", return_value=None):
        assert fn() is False

    with patch("importlib.util.find_spec", return_value=True):
        assert fn() is True


def test_is_hopper_gpu_available():
    with patch("torch.cuda.is_available", return_value=False):
        assert is_hopper_gpu_available() is False

    with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
        with patch("torch.cuda.is_available", return_value=True):
            assert is_hopper_gpu_available() is True

    with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
        with patch("torch.cuda.is_available", return_value=True):
            assert is_hopper_gpu_available() is False


def test_is_torch_gpu_available():
    with patch("torch.cuda.is_available", return_value=False):
        assert is_torch_gpu_available() is False

    with patch("torch.cuda.is_available", return_value=True):
        assert is_torch_gpu_available() is True
