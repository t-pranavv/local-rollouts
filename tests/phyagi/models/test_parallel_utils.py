# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from phyagi.models.parallel_utils import (
    ShardColwiseParallel,
    clip_grad_by_total_norm_,
    get_grad_norm,
    maybe_apply_context_parallel_to_inputs,
)


class DummyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)


def _make_mock_meshes():
    dp_mesh = MagicMock()
    tp_mesh = MagicMock()
    dp_mesh.get_group.return_value = "dp_group"
    tp_mesh.get_group.return_value = "tp_group"
    return dp_mesh, tp_mesh


@patch("phyagi.models.parallel_utils.distribute_tensor")
def test_shard_colwise_parallel(distribute_mock):
    in_features, out_features = 4, 8
    shard_sizes_valid = [4, 4]
    shard_sizes_invalid = [2, 2]

    model = DummyLinear(in_features, out_features)
    original_weight = model.linear.weight.data.clone()
    device_mesh = MagicMock()
    device_mesh.size.return_value = 2

    dummy_sharded_tensor = torch.zeros_like(model.linear.weight.data)
    distribute_mock.return_value = dummy_sharded_tensor

    strategy = ShardColwiseParallel(shard_sizes=shard_sizes_valid)
    strategy._partition_linear_fn("linear", model.linear, device_mesh)

    distribute_mock.assert_called_once()

    expected_order = torch.cat(
        [
            original_weight[0:2],
            original_weight[4:6],
            original_weight[2:4],
            original_weight[6:8],
        ],
        dim=0,
    )
    torch.testing.assert_close(distribute_mock.call_args[0][0], expected_order)

    strategy_invalid = ShardColwiseParallel(shard_sizes=shard_sizes_invalid)
    with pytest.raises(ValueError):
        strategy_invalid._partition_linear_fn("linear", model.linear, device_mesh)


def test_maybe_apply_context_parallel_to_inputs_basic_partitioning():
    inputs = {
        "input_ids": torch.arange(12).reshape(2, 6),
        "attention_mask": torch.ones(2, 6),
    }
    result = maybe_apply_context_parallel_to_inputs(inputs.copy(), 2, 1)

    assert result["input_ids"].shape == (2, 3)
    assert torch.equal(result["input_ids"], torch.tensor([[3, 4, 5], [9, 10, 11]]))
    assert "position_ids" in result
    assert torch.equal(result["position_ids"], torch.tensor([[3, 4, 5]]))


@patch("torch.distributed.all_reduce")
@pytest.mark.is_torch_gpu
def test_get_grad_norm_l2(mock_all_reduce):
    model = nn.Linear(4, 4)
    dp_mesh, tp_mesh = _make_mock_meshes()

    for p in model.parameters():
        p.grad = torch.ones_like(p)

    norm = get_grad_norm(list(model.parameters()), dp_mesh, tp_mesh, norm_type=2.0)
    expected = (20.0) ** 0.5

    assert pytest.approx(norm, rel=1e-4) == expected
    assert mock_all_reduce.call_count == 2


@patch("torch.distributed.all_reduce")
@pytest.mark.is_torch_gpu
def test_clip_grad_by_total_norm_(mock_all_reduce):
    model = nn.Linear(4, 4)

    for p in model.parameters():
        p.grad = torch.full_like(p, 2.0)

    grads_before = [p.grad.clone() for p in model.parameters()]
    total_norm = get_grad_norm(list(model.parameters()), *_make_mock_meshes(), norm_type=2.0)

    clip_grad_by_total_norm_(list(model.parameters()), max_norm=1.0, total_norm=total_norm)

    for grad_before, p in zip(grads_before, model.parameters()):
        expected = grad_before * (1.0 / (total_norm + 1e-6))
        torch.testing.assert_close(p.grad, expected)
