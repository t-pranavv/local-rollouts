# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ConstantLR, LRScheduler

from phyagi.optimizers.registry import get_lr_scheduler, get_optimizer


def test_get_optimizer():
    model = torch.nn.Linear(in_features=10, out_features=5)

    optimizer_type = "adamw"
    optimizer_kwargs = {"lr": 0.001, "weight_decay": 0.01}

    optimizer = get_optimizer(model, optimizer_type, **optimizer_kwargs)
    assert isinstance(optimizer, (Optimizer, AdamW))
    assert optimizer.defaults["lr"] == optimizer_kwargs["lr"]
    assert optimizer.defaults["weight_decay"] == optimizer_kwargs["weight_decay"]


def test_get_lr_scheduler():
    model = torch.nn.Linear(in_features=10, out_features=5)
    optimizer = AdamW(model.parameters(), lr=0.001)

    lr_scheduler_type = "constant"
    lr_scheduler_kwargs = {"total_iters": 5}

    lr_scheduler = get_lr_scheduler(optimizer, lr_scheduler_type, **lr_scheduler_kwargs)
    assert isinstance(lr_scheduler, (LRScheduler, ConstantLR))
    assert lr_scheduler.total_iters == lr_scheduler_kwargs["total_iters"]
