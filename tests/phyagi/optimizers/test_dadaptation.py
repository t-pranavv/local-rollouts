# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.nn import Parameter

from phyagi.optimizers.dadaptation import DAdaptAdam


def test_dadapt_adam():
    model_params = [Parameter(torch.randn(10, 10)), Parameter(torch.randn(10))]

    optimizer = DAdaptAdam(model_params)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 1.0
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["d"] == 1e-6

    loss = torch.tensor(10.0)

    def closure():
        return loss

    result = optimizer.step(closure)
    assert result == loss

    result = optimizer.step()
    assert result is None
