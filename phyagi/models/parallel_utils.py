# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from itertools import accumulate
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel


class ShardColwiseParallel(ColwiseParallel):
    """A column-wise partitioning strategy for ``nn.Linear`` layers, enabling sharding along the
    output dimension based on a predefined list of shard sizes.

    Unlike PyTorch's default ``ColwiseParallel``, this implementation correctly handles packed
    weight tensors, such as those used in Gated Linear Units (GLUs), where multiple projections
    (e.g., gate and up projection) are stored in a single weight matrix.

    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
        shard_sizes: List[int] = None,
    ) -> None:
        super().__init__(
            input_layouts=input_layouts,
            output_layouts=output_layouts,
            use_local_output=use_local_output,
        )

        if shard_sizes is None:
            raise ValueError("`shard_sizes` must be defined, but got None.")
        self._shard_sizes = shard_sizes

    def _partition_linear_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
        world_size = device_mesh.size(0)

        for name, param in module.named_parameters():
            if sum(self._shard_sizes) != param.size(0):
                raise ValueError(f"Sum of `shard_sizes` must be {param.size(0)}, but got {sum(self._shard_sizes)}.")

            # Calculate local shard sizes for each shard
            local_shard_sizes = [size // world_size for size in self._shard_sizes]

            # Calculate cumulative sizes for each shard
            cumulative_sizes = [0] + list(accumulate(self._shard_sizes))
            cumulative_local_sizes = [0] + list(accumulate(local_shard_sizes))

            shard_param = torch.empty_like(param)
            for i in range(world_size):
                for j, shard_size in enumerate(self._shard_sizes):
                    local_shard_size = shard_size // world_size

                    # Source indexes (based on original layout)
                    src_start = cumulative_sizes[j] + i * local_shard_size
                    src_end = src_start + local_shard_size

                    # Target indices (interleaved, based on cumulative local sizes)
                    tgt_start = i * sum(local_shard_sizes) + cumulative_local_sizes[j]
                    tgt_end = tgt_start + local_shard_size

                    shard_param[tgt_start:tgt_end] = param[src_start:src_end].detach().clone()

            dist_param = distribute_tensor(shard_param, device_mesh, [Shard(0)])
            module.register_parameter(name, nn.Parameter(dist_param))

    @staticmethod
    def unshard_param(
        param: Union[torch.Tensor, DTensor],
        shard_sizes: List[int],
        device_mesh: DeviceMesh,
    ) -> torch.Tensor:
        """Unshard a parameter tensor based on the provided shard sizes and device mesh.

        Args:
            param: Tensor to be unsharded.
            shard_sizes: List of sizes for each shard.
            device_mesh: Device mesh that defines the tensor parallelism.

        Returns:
            A tensor that is the concatenation of the shards, or the original tensor if it cannot be unsharded.

        """

        # Skip if the parameter is on a meta device
        if getattr(param, "device", None) is not None and param.device.type == "meta":
            return param

        world_size = device_mesh.size(0)
        if isinstance(param, DTensor):
            param = param.full_tensor().cpu()

        n_total, _ = param.shape
        if n_total != sum(shard_sizes) or world_size == 1:
            return param

        per_rank_piece = []
        for s in shard_sizes:
            if s % world_size:
                return param
            per_rank_piece.append(s // world_size)

        shards = torch.chunk(param, world_size, dim=0)

        offsets = [0]
        for sz in per_rank_piece:
            offsets.append(offsets[-1] + sz)

        pieces = []
        for i in range(len(shard_sizes)):
            piece = torch.cat([shard[offsets[i] : offsets[i + 1]] for shard in shards], dim=0)
            pieces.append(piece)

        return torch.cat(pieces, dim=0)


def _create_context_parallel_position_ids(
    inputs: Dict[str, torch.Tensor], start_idx: int, end_idx: int
) -> torch.Tensor:
    return torch.arange(
        start_idx, end_idx, dtype=torch.long, device=inputs["input_ids"].device if "input_ids" in inputs else None
    ).unsqueeze(0)


def _maybe_create_context_parallel_cu_seqlens(
    inputs: Dict[str, torch.Tensor], start_idx: int, end_idx: int, seq_len: int
) -> Optional[torch.Tensor]:
    if "cu_seqlens" not in inputs:
        return None
    if "boundaries" not in inputs:
        raise ValueError("Splitting `cu_seqlens` requires `boundaries` to be available in `inputs`.")

    flattened = []
    offset = 0
    for boundaries in inputs["boundaries"]:
        local_bounds = [int(b - start_idx) for b in boundaries if start_idx <= b <= end_idx]

        if not local_bounds or local_bounds[0] != 0:
            local_bounds.insert(0, 0)
        if local_bounds[-1] != seq_len:
            local_bounds.append(seq_len)

        flattened.extend([b + offset for b in local_bounds])
        offset += seq_len

    return torch.tensor(
        flattened, dtype=torch.int32, device=inputs["input_ids"].device if "input_ids" in inputs else None
    )


def maybe_apply_context_parallel_to_inputs(
    inputs: Dict[str, torch.Tensor], context_parallel_world_size: int = 1, context_parallel_rank: int = 1
) -> Dict[str, torch.Tensor]:
    """Applies context parallelism to the input tensors in the given dictionary.

    This function modifies the input tensors in place, slicing them according to the context parallel
    rank and world size.

    Special keys, such as ``position_ids`` and ``cu_seqlens``, are handled separately to ensure
    that they are correctly adjusted for context parallelism.

    Args:
        inputs: Dictionary of input tensors, where each tensor is expected to have the same sequence length.
        context_parallel_world_size: Size of the context parallel.
        context_parallel_rank: Rank of the current process in the context parallel.

    Returns:
        Dictionary of input tensors, sliced according to the context parallel rank and world size.

    """

    CONTEXT_PARALLEL_SPECIAL_KEYS = {"position_ids", "cu_seqlens"}
    CONTEXT_PARALLEL_IGNORE_KEYS = {"valid_seqlens", "boundaries"}

    if context_parallel_world_size <= 1:
        return inputs

    seq_lengths = {
        k: v.size(1)
        for k, v in inputs.items()
        if isinstance(v, torch.Tensor)
        and k not in CONTEXT_PARALLEL_SPECIAL_KEYS
        and k not in CONTEXT_PARALLEL_IGNORE_KEYS
    }
    if len(set(seq_lengths.values())) != 1:
        raise ValueError(
            f"`context_parallel_world_size > 1` requires all tensors to have the same sequence length, but got {seq_lengths}."
        )

    seq_length = next(iter(seq_lengths.values()))

    sub_seq_length = seq_length // context_parallel_world_size
    sub_seq_start = context_parallel_rank * sub_seq_length
    sub_seq_end = (context_parallel_rank + 1) * sub_seq_length

    # Iterate over the input tensors and slice them according to the context parallel rank
    for k, v in inputs.items():
        if k in CONTEXT_PARALLEL_SPECIAL_KEYS or k in CONTEXT_PARALLEL_IGNORE_KEYS:
            continue
        if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.size(1) == seq_length:
            inputs[k] = v[:, sub_seq_start:sub_seq_end]

    # Special keys are handled separately to ensure they are correctly adjusted for context parallelism
    inputs["position_ids"] = _create_context_parallel_position_ids(inputs, sub_seq_start, sub_seq_end)
    inputs["cu_seqlens"] = _maybe_create_context_parallel_cu_seqlens(inputs, sub_seq_start, sub_seq_end, sub_seq_length)

    return inputs


def _convert_tensor_to_local(tensor: Union[torch.Tensor, DTensor]) -> torch.Tensor:
    with torch.no_grad():
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def get_grad_norm(
    parameters: Union[list[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]],
    dp_mesh: DeviceMesh,
    tp_mesh: DeviceMesh,
    norm_type: Union[int, float] = 2,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Calculate the norm of gradients.

    Reference:
        https://github.com/NVIDIA/Megatron-LM/blob/a695b2bd2a0ca9ca63385a48c41a1c5a033cdd1e/megatron/core/optimizer/clip_grads.py#L51

    Args:
        parameters: Iterable of parameters to clip.
        dp_mesh: Data parallel device mesh.
        tp_mesh: Tensor parallel device mesh.
        norm_type: Type of norm to use (default is 2.0).
        dtype: Data type to use for the gradients.

    Returns:
        Total norm of the gradients.

    """

    dp_group = dp_mesh.get_group()
    tp_group = tp_mesh.get_group()

    if isinstance(parameters, (torch.Tensor, DTensor)):
        parameters = [parameters]

    grads = [_convert_tensor_to_local(p.grad.detach()).to(dtype) for p in parameters if p.grad is not None]

    norm_type = float(norm_type)
    if norm_type == torch.inf:
        total_norm = max(grad.abs().max().item() for grad in grads)
        total_norm = torch.tensor([float(total_norm)], dtype=torch.float, device="cuda")

        torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=dp_group)
        torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=tp_group)

        total_norm = total_norm[0].item()
    else:
        total_norm = torch.tensor(0.0, dtype=dtype)

        for grad in grads:
            grad_norm = torch.norm(grad, norm_type)
            total_norm = total_norm.to(grad_norm.device) + grad_norm**norm_type

        total_norm = total_norm.cuda()

        torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        total_norm = total_norm.item() ** (1.0 / norm_type)

    return total_norm


def clip_grad_by_total_norm_(
    parameters: Union[list[Union[torch.Tensor, DTensor]], Union[torch.Tensor, DTensor]],
    max_norm: Union[int, float],
    total_norm: float,
    dtype: torch.dtype = torch.float32,
) -> None:
    """Clip gradients by total norm and modify them in place.

    Reference:
        https://github.com/NVIDIA/Megatron-LM/blob/a695b2bd2a0ca9ca63385a48c41a1c5a033cdd1e/megatron/core/optimizer/clip_grads.py#L138

    Args:
        parameters: Iterable of parameters to clip.
        max_norm: Maximum norm value.
        total_norm: Pre-computed total norm of the gradients to use for scaling.
        dtype: Data type to use for the gradients.

    """

    if isinstance(parameters, (torch.Tensor, DTensor)):
        parameters = [parameters]

    grads = [_convert_tensor_to_local(p.grad.detach()).to(dtype) for p in parameters if p.grad is not None]

    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        for g in grads:
            g.mul_(clip_coeff)
