# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time

from phyagi.utils.timer import Timer


def test_timer():
    timer = Timer()

    timer.start()
    time.sleep(0.01)

    metrics = timer.end()
    assert "total" in metrics
    assert metrics["total"] > 0

    timer = Timer()
    with timer.measure("step1"):
        time.sleep(0.01)

    metrics = timer.metrics()
    assert "step1" in metrics
    assert metrics["step1"] > 0
