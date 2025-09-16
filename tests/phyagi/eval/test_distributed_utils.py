# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

torch.distributed = pytest.importorskip("torch.distributed")

from unittest import mock

from phyagi.eval.distributed_utils import (
    all_gather_list,
    all_reduce_dict,
    get_rank,
    get_world_size,
    is_main_local_process,
    is_main_process,
)


def test_get_rank():
    assert get_rank() == 0


def test_get_world_size():
    assert get_world_size() == 1


def test_is_main_process():
    assert is_main_process() is True


def test_is_main_local_process():
    assert is_main_local_process() is True


def test_all_reduce_dict():
    input_dict = {"a": 1, "b": 2}
    world_size = 2

    with mock.patch("torch.distributed.is_initialized", return_value=True), mock.patch(
        "torch.distributed.get_world_size", return_value=world_size
    ), mock.patch("torch.distributed.all_reduce") as mock_all_reduce:

        def mock_reduce(tensor, op):
            tensor *= world_size

        mock_all_reduce.side_effect = mock_reduce
        result = all_reduce_dict(input_dict)

    expected_result = {key: value * world_size / world_size for key, value in input_dict.items()}
    assert result == expected_result


def test_all_gather_list():
    input_list = [1, 2, 3]
    world_size = 2
    gathered_data = [[1, 2, 3], [4, 5, 6]]

    with mock.patch("torch.distributed.is_initialized", return_value=True), mock.patch(
        "torch.distributed.get_world_size", return_value=world_size
    ), mock.patch("torch.distributed.all_gather_object") as mock_all_gather:

        def mock_gather(output_list, input_obj):
            output_list[:] = gathered_data

        mock_all_gather.side_effect = mock_gather
        result = all_gather_list(input_list)

    expected_result = [item for sublist in gathered_data for item in sublist]
    assert result == expected_result
