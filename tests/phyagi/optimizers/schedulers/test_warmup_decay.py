# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch
from torch.optim import SGD

from phyagi.optimizers.schedulers.warmup_decay import (
    WarmupDecayCooldownLR,
    WarmupDecayLR,
    WarmupLR,
)


@pytest.fixture
def optimizer():
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    return SGD(model, 0.1)


def test_warmup_lr(optimizer):
    scheduler = WarmupLR(optimizer, warmup_num_steps=1000, warmup_max_lr=0.01)

    for _ in range(1000):
        optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr()[0] == pytest.approx(0.01, rel=1e-4)


def test_warmup_lr_log(optimizer):
    scheduler = WarmupLR(optimizer, warmup_num_steps=1000, warmup_max_lr=0.01, warmup_type="log")

    for _ in range(1000):
        optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr()[0] == pytest.approx(0.01, rel=1e-4)


def test_warmup_lr_invalid_type(optimizer):
    with pytest.raises(ValueError):
        WarmupLR(optimizer, warmup_type="invalid")


def test_warmup_decay_lr(optimizer):
    scheduler = WarmupDecayLR(optimizer, total_num_steps=10000, warmup_num_steps=1000, warmup_max_lr=0.01)

    for _ in range(10000):
        optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr()[0] == pytest.approx(1.1e-6, abs=1e-6)


def test_warmup_decay_lr_log(optimizer):
    scheduler = WarmupDecayLR(
        optimizer, total_num_steps=10000, warmup_num_steps=1000, warmup_max_lr=0.01, warmup_type="log"
    )

    for _ in range(10000):
        optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr()[0] == pytest.approx(1.1e-6, abs=1e-6)


def test_warmup_decay_lr_invalid_steps(optimizer):
    with pytest.raises(ValueError):
        WarmupDecayLR(optimizer, total_num_steps=500, warmup_num_steps=1000)


def test_warmup_decay_cooldown_lr(optimizer):
    scheduler = WarmupDecayCooldownLR(optimizer, total_num_steps=10000)

    for _ in range(10000):
        optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr()[0] == 0.001


def test_warmup_decay_cooldown_lr_log(optimizer):
    scheduler = WarmupDecayCooldownLR(optimizer, total_num_steps=10000, warmup_type="log")

    for _ in range(1000):
        optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr()[0] == 0.001
