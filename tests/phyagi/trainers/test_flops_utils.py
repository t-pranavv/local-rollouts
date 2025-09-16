# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math

import pytest

from phyagi.trainers.flops_utils import estimate_tflops, get_peak_tflops


def test_get_peak_tflops():
    assert get_peak_tflops("INVALID") == math.inf
    assert get_peak_tflops("a6000") == 309.7
    assert get_peak_tflops("A6000") == 309.7
    assert get_peak_tflops("A100") == 312.0
    assert get_peak_tflops("H100 NVL") == 835.0
    assert get_peak_tflops("H100 PCIe") == 756.0
    assert get_peak_tflops("H100 PCIE") == 756.0
    assert get_peak_tflops("H100") == 989.0
    assert get_peak_tflops("H200") == 989.0
    assert get_peak_tflops("MI300X") == 1300.0
    assert get_peak_tflops("MI325X") == 1300.0
    assert get_peak_tflops("MI250X") == 191.5


def test_estimate_tflops():
    n_layer = 12
    n_embd = 768
    vocab_size = 30522
    seq_len = 1024
    batch_size = 2
    step_time = 0.001

    result = estimate_tflops(step_time, n_layer, n_embd, vocab_size, seq_len, batch_size)
    expected_result = (
        (24 * 3 * batch_size * seq_len * n_layer * (n_embd**2))
        * (1 + (seq_len / (6 * n_embd)) + (vocab_size / (16 * n_layer * n_embd)))
        / (step_time * 1e12)
    )
    assert result == pytest.approx(expected_result)
