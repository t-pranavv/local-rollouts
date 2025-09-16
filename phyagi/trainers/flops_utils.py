# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import Optional

import torch


def get_peak_tflops(device_name: Optional[str] = None) -> float:
    """Get the peak TFLOPs for a given device.

    Args:
        device_name: Device name.

    Returns:
        Peak TFLOPs.

    """

    if device_name is None and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()

    if device_name is None:
        return math.inf

    device_name = device_name.upper()

    # https://www.leadtek.com/eng/products/workstation_graphics(2)/NVIDIA_RTX_A6000(30893)/detail
    if "A6000" in device_name:
        return 309.7

    # https://www.nvidia.com/en-us/data-center/a100
    if "A100" in device_name:
        return 312.0

    # https://www.nvidia.com/en-us/data-center/h100
    if "H100" in device_name:
        if "NVL" in device_name:
            return 835.0
        if "PCIE" in device_name:
            return 756.0
        # SXM and other variants
        return 989.0

    # https://www.nvidia.com/en-us/data-center/h200
    if "H200" in device_name:
        return 989.0

    # https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
    # https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
    if "MI300X" in device_name or "MI325X" in device_name:
        return 1300.0

    # https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html
    if "MI250X" in device_name:
        return 191.5

    return math.inf


def estimate_tflops(
    step_time: float,
    n_layer: int = 1,
    n_embd: int = 1,
    vocab_size: int = 1,
    seq_len: int = 1,
    batch_size: int = 1,
    activation_checkpointing: bool = False,
) -> float:
    """Estimate the number of TFLOPs (equation 3, section 5.1).

    Since the estimation is based on the ``step_time``, the result might
    comprehend multiple GPUs and/or multiple nodes if the ``step_time``
    is the total time for the iteration.

    Reference:
        Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.
        https://arxiv.org/pdf/2104.04473.pdf

    Args:
        step_time: Time per step (in seconds).
        n_layer: Number of layers.
        n_embd: Embedding dimension.
        vocab_size: Vocabulary size.
        seq_len: Sequence length.
        batch_size: Batch size.
        activation_checkpointing: Whether activation checkpointing is being used.

    Returns:
        Number of TFLOPs.

    """

    act_ckp_factor = 4 if activation_checkpointing else 3
    flops_per_iteration = (24 * act_ckp_factor * batch_size * seq_len * n_layer * (n_embd**2)) * (
        1 + (seq_len / (6 * n_embd)) + (vocab_size / (16 * n_layer * n_embd))
    )

    return flops_per_iteration / (step_time * 1e12)
