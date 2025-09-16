# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

from phyagi.optimizers.lion import Lion


def test_lion_optimizer():
    model = torch.nn.Linear(2, 1)
    params = model.parameters()
    optimizer = Lion(params)

    assert optimizer.defaults["lr"] == 1e-4
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["weight_decay"] == 0.0

    with pytest.raises(ValueError):
        Lion(params, lr=-1e-4)
    with pytest.raises(ValueError):
        Lion(params, betas=(0.9, 1.1))

    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    targets = torch.tensor([[3.0], [7.0]])
    loss_fn = torch.nn.MSELoss()

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()

    updated_params = list(model.parameters())
    assert len(updated_params) == 2
    assert not torch.equal(updated_params[0], torch.zeros_like(updated_params[0]))
    assert not torch.equal(updated_params[1], torch.zeros_like(updated_params[1]))
