# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from phyagi.utils.import_utils import (
    is_flash_attn_available,
    is_mpi_available,
    is_torch_gpu_available,
    is_vllm_available,
)


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False, help="Run slow tests.")
    parser.addoption("--slowest", action="store_true", default=False, help="Run the slowest tests.")


def pytest_configure(config):
    config.addinivalue_line("markers", "is_flash_attn: requires Flash-Attention.")
    config.addinivalue_line("markers", "is_mpi: requires MPI.")
    config.addinivalue_line("markers", "is_torch_gpu: requires PyTorch (GPU).")
    config.addinivalue_line("markers", "is_vllm: requires VLLM.")
    config.addinivalue_line("markers", "slow: runs slowly (more than 1 minute and less than 5 minutes).")
    config.addinivalue_line("markers", "slowest: runs super slowly (more than 5 minutes).")


def pytest_runtest_setup(item):
    is_flash_attn = item.get_closest_marker("is_flash_attn")
    if is_flash_attn and not is_flash_attn_available():
        pytest.skip("Flash-Attention is not available.")

    is_mpi = item.get_closest_marker("is_mpi")
    if is_mpi and not is_mpi_available():
        pytest.skip("MPI is not available.")

    is_torch_gpu = item.get_closest_marker("is_torch_gpu")
    if is_torch_gpu and not is_torch_gpu_available():
        pytest.skip("PyTorch (GPU) is not available.")

    is_vllm = item.get_closest_marker("is_vllm")
    if is_vllm and not is_vllm_available():
        pytest.skip("vLLM is not available.")

    slow = item.get_closest_marker("slow")
    if slow and not item.config.getoption("--slow"):
        pytest.skip("Slow tests are not enabled.")

    slowest = item.get_closest_marker("slowest")
    if slowest and not item.config.getoption("--slowest"):
        pytest.skip("Slowest tests are not enabled.")
