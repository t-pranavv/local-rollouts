# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import torch
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh

from phyagi.models.mixformer_sequential.modeling_mixformer_sequential import (
    MixFormerSequentialForCausalLM,
    MixFormerSequentialForSequenceClassification,
)
from phyagi.utils.logging_utils import get_logger

logger = get_logger(__name__)


def apply_ac_mixformer_sequential(
    model: Union[MixFormerSequentialForCausalLM, MixFormerSequentialForSequenceClassification],
    preserve_rng_state: bool = False,
) -> None:
    """Apply Activation Checkpointing (AC) to a :class:`MixFormerSequentialForCausalLM` or
    :class:`MixFormerSequentialForSequenceClassification`

    Args:
        model: Input model.
        preserve_rng_state: Whether to preserve the random number generator state.

    """

    if getattr(model.config, "gradient_checkpointing", False):
        raise ValueError("`gradient_checkpointing` must be False when using `apply_ac_mixformer_sequential()`.")

    n_blocks = len(model.layers[1:-1])
    for block_id in range(n_blocks):
        model.layers[block_id + 1] = checkpoint_wrapper(
            model.layers[block_id + 1], preserve_rng_state=preserve_rng_state
        )
    logger.info("Activation Checkpointing (AC) has been applied.")


def apply_cp_mixformer_sequential(
    model: Union[MixFormerSequentialForCausalLM, MixFormerSequentialForSequenceClassification],
    cp_mesh: DeviceMesh,
    varlen: bool = False,
) -> None:
    """Apply Context Parallelism (CP) to a :class:`MixFormerSequentialForCausalLM` or
    :class:`MixFormerSequentialForSequenceClassification` model.

    Args:
        model: Input model.
        cp_mesh: Context parallelism mesh.
        varlen: Whether to use variable-length sequences.
            If ``True``, the input sequence must be evenly divisible by the context parallel size.

    """

    if cp_mesh.size() <= 1:
        return

    def _maybe_set_cp(module: torch.nn.Module) -> None:
        if hasattr(module, "set_torch_cp"):
            module.set_torch_cp(cp_mesh, varlen=varlen)

    model.apply(_maybe_set_cp)
    logger.info("Context Parallelism (CP) has been applied to supported modules.")


def apply_tp_mixformer_sequential(
    model: Union[MixFormerSequentialForCausalLM, MixFormerSequentialForSequenceClassification],
    tp_mesh: DeviceMesh,
    enable_async: bool = False,
    enable_sequence_parallel: bool = False,
    enable_loss_parallel: bool = False,
) -> None:
    """Apply Tensor Parallelism (TP) to a :class:`MixFormerSequentialForCausalLM` or
    :class:`MixFormerSequentialForSequenceClassification` model.

    Args:
        model: Input model.
        tp_mesh: Tensor parallelism mesh.
        enable_async: Whether to use asynchronous communication.
        enable_sequence_parallel: Whether to use sequence parallel.
            If ``True``, the input sequence must be evenly divisible by the tensor parallel size.
        enable_loss_parallel: Whether to use loss parallel.

    """

    if tp_mesh.size() <= 1:
        return

    def _maybe_set_tp(module: torch.nn.Module) -> None:
        if hasattr(module, "set_torch_tp"):
            module.set_torch_tp(
                tp_mesh, enable_sequence_parallel=enable_sequence_parallel, enable_loss_parallel=enable_loss_parallel
            )

    model.apply(_maybe_set_tp)
    logger.info("Tensor Parallelism (TP) has been applied to supported modules.")

    if enable_async:
        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)
        logger.info("Asynchronous Tensor Parallelism (TP) has been applied.")


def apply_fsdp_mixformer_sequential(
    model: Union[MixFormerSequentialForCausalLM, MixFormerSequentialForSequenceClassification],
    dp_mesh: DeviceMesh,
    precision: str,
    compile: bool = False,
    cpu_offload: bool = False,
) -> None:
    """Apply Fully Sharded Data Parallelism (FSDP) to a :class:`MixFormerSequentialForCausalLM` or
    :class:`MixFormerSequentialForSequenceClassification` model.

    Args:
        model: Input model.
        dp_mesh: Data parallelism mesh.
        precision: Training precision.
        compile: Whether to compile the model.
        cpu_offload: Whether to offload to CPU.

    """

    if compile:
        for layer_id in range(len(model.layers[1:-1])):
            model.layers[layer_id + 1] = torch.compile(model.layers[layer_id + 1])
        logger.info("`torch.compile()` has been applied.")

    if dp_mesh.size() <= 1:
        return

    param_dtype = torch.float32
    if "bf16" in precision or "bfloat16" in precision:
        param_dtype = torch.bfloat16
    elif "16" in precision or "float16" in precision:
        param_dtype = torch.float16

    fsdp_config = {
        "mesh": dp_mesh,
        "mp_policy": MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32),
        "offload_policy": CPUOffloadPolicy() if cpu_offload else None,
    }

    n_blocks = len(model.layers[1:-1])
    for block_id in range(n_blocks):
        block = model.layers[block_id + 1]
        if hasattr(block, "set_torch_fsdp"):
            block.set_torch_fsdp(dp_mesh, fsdp_config, block_id + 1, n_blocks)
    logger.info("Fully Sharded Data Parallelism (FSDP) has been applied to model blocks.")

    fully_shard(model, **fsdp_config)
    logger.info("Fully Sharded Data Parallelism (FSDP) has been applied to model.")
