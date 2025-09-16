# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from phyagi.trainers.pl.pl_strategies import DataContextTensorParallelStrategy


@pytest.fixture(autouse=True)
def setup_distributed_env():
    original_environ = dict(os.environ)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"

    yield

    os.environ.clear()
    os.environ.update(original_environ)


@pytest.fixture
def strategy():
    return DataContextTensorParallelStrategy(data_parallel_size=1, tensor_parallel_size=1)


@patch("torch.distributed.device_mesh.DeviceMesh")
@patch("torch.distributed.init_process_group")
@patch("torch.distributed.destroy_process_group")
@pytest.mark.parametrize(
    "data_size,context_size, tensor_size,world_size,should_raise",
    [
        (1, 1, 1, 1, False),
        (2, 1, 1, 2, False),
        (2, 1, 2, 4, False),
        (2, 1, 2, 6, True),
    ],
)
def test_setup_device_mesh(
    mock_destroy_pg, mock_init_pg, mock_device_mesh, data_size, context_size, tensor_size, world_size, should_raise
):
    from phyagi.trainers.pl.pl_strategies import _setup_device_mesh

    if should_raise:
        with pytest.raises(ValueError):
            _setup_device_mesh(
                data_size,
                context_size,
                tensor_size,
                world_size,
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
    else:
        mock_device_mesh.return_value = MagicMock()

        _ = _setup_device_mesh(
            data_size,
            context_size,
            tensor_size,
            world_size,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        mock_device_mesh.assert_called_once()


def test_strategy_initialization():
    strategy = DataContextTensorParallelStrategy(
        data_parallel_size=2, context_parallel_size=1, tensor_parallel_size=1, cpu_offload=True
    )

    assert strategy._data_parallel_size == 2
    assert strategy._context_parallel_size == 1
    assert strategy._tensor_parallel_size == 1
    assert strategy._cpu_offload is True
    assert strategy.process_group_backend == "cuda:nccl,cpu:gloo"
