# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.distributed import ProcessGroup

_DATA_PARALLEL_GLOBAL_RANKS = None
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_DATA_PARALLEL_GROUP = None


def initialize(context_parallel_size: int = 1, use_distributed_optimizer: bool = False) -> None:
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before calling `initialize()`.")

    world_size = torch.distributed.get_world_size()
    if world_size % context_parallel_size != 0:
        raise ValueError(
            f"`world_size` must be divisible by `context_parallel_size`, but got {world_size} and {context_parallel_size}."
        )

    data_parallel_size = world_size // context_parallel_size
    context_data_parallel_size = context_parallel_size * data_parallel_size

    n_context_parallel_groups = world_size // context_parallel_size
    n_context_data_parallel_groups = world_size // context_parallel_size // data_parallel_size

    rank = torch.distributed.get_rank()

    # Data parallel groups
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    if _DATA_PARALLEL_GROUP is not None:
        raise RuntimeError("`data_parallel` group is already initialized.")

    data_parallel_group_ranks = []
    for j in range(context_parallel_size):
        ranks = range(j, world_size, context_parallel_size)

        data_parallel_group_ranks.append(list(ranks))
        group = torch.distributed.new_group(ranks)
        group_gloo = torch.distributed.new_group(ranks, backend="gloo") if use_distributed_optimizer else None

        if rank in ranks:
            _DATA_PARALLEL_GROUP = group
            _DATA_PARALLEL_GROUP_GLOO = group_gloo
            _DATA_PARALLEL_GLOBAL_RANKS = ranks

    # Context parallel groups
    global _CONTEXT_PARALLEL_GROUP
    if _CONTEXT_PARALLEL_GROUP is not None:
        raise RuntimeError("`context_parallel` group is already initialized.")

    for i in range(n_context_parallel_groups):
        ranks = range(i * context_parallel_size, (i + 1) * context_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group

    # Context-data parallel groups
    global _CONTEXT_DATA_PARALLEL_GROUP
    if _CONTEXT_DATA_PARALLEL_GROUP is not None:
        raise RuntimeError("`context_data_parallel` group is already initialized.")

    data_context_parallel_group_ranks = []
    for i in range(n_context_data_parallel_groups):
        ranks = range(i * context_data_parallel_size, (i + 1) * context_data_parallel_size)

        data_context_parallel_group_ranks.append(list(ranks))
        group = torch.distributed.new_group(ranks)

        if rank in ranks:
            _CONTEXT_DATA_PARALLEL_GROUP = group


def get_data_parallel_group() -> ProcessGroup:
    if _DATA_PARALLEL_GROUP is None:
        raise RuntimeError("`data_parallel` group has not been initialized yet.")
    return _DATA_PARALLEL_GROUP


def get_data_parallel_group_gloo() -> ProcessGroup:
    if _DATA_PARALLEL_GROUP_GLOO is None:
        raise RuntimeError("`data_parallel_gloo` group has not been initialized yet.")
    return _DATA_PARALLEL_GROUP_GLOO


def get_data_parallel_world_size() -> int:
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank() -> int:
    return torch.distributed.get_rank(group=get_data_parallel_group())


def get_sequence_parallel_group() -> ProcessGroup:
    if _CONTEXT_PARALLEL_GROUP is None:
        raise RuntimeError("`context_parallel` group has not been initialized yet.")
    return _CONTEXT_PARALLEL_GROUP


def get_sequence_parallel_world_size() -> int:
    return torch.distributed.get_world_size(group=get_sequence_parallel_group())


def get_sequence_parallel_rank() -> int:
    return torch.distributed.get_rank(group=get_sequence_parallel_group())


def get_sequence_data_parallel_group() -> ProcessGroup:
    if _CONTEXT_DATA_PARALLEL_GROUP is None:
        raise RuntimeError("`context_data_parallel` group has not been initialized yet.")
    return _CONTEXT_DATA_PARALLEL_GROUP


def get_sequence_data_parallel_world_size() -> int:
    return torch.distributed.get_world_size(group=get_sequence_data_parallel_group())


def get_sequence_data_parallel_rank() -> int:
    return torch.distributed.get_rank(group=get_sequence_data_parallel_group())


def get_model_parallel_group() -> None:
    return None


def get_model_parallel_world_size() -> int:
    return 1


def get_model_parallel_rank() -> int:
    return 0
