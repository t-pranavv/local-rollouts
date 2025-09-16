# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import deepspeed
import pytest
import torch.distributed as dist

from phyagi.trainers.ds import ds_mpu


@pytest.mark.is_mpi
@pytest.mark.is_torch_gpu
def test_initialize_and_get_groups():
    deepspeed.init_distributed()

    world_size = dist.get_world_size()
    context_parallel_size = 1
    use_distributed_optimizer = True

    ds_mpu.initialize(context_parallel_size, use_distributed_optimizer)

    # Data Parallel group
    dp_group = ds_mpu.get_data_parallel_group()
    dp_world_size = dist.get_world_size(dp_group)
    dp_rank = dist.get_rank(dp_group)
    assert dp_world_size == world_size // context_parallel_size
    assert 0 <= dp_rank < dp_world_size

    # Data parallel group (Gloo)
    dp_group_gloo = ds_mpu.get_data_parallel_group_gloo()
    dp_world_size_gloo = dist.get_world_size(dp_group_gloo)
    dp_rank_gloo = dist.get_rank(dp_group_gloo)
    assert dp_world_size_gloo == dp_world_size
    assert dp_rank_gloo == dp_rank

    # Sequence parallel group
    seq_group = ds_mpu.get_sequence_parallel_group()
    seq_world_size = dist.get_world_size(seq_group)
    seq_rank = dist.get_rank(seq_group)
    assert seq_world_size == context_parallel_size
    assert 0 <= seq_rank < seq_world_size

    # Sequence-Data parallel group
    sdp_group = ds_mpu.get_sequence_data_parallel_group()
    sdp_world_size = dist.get_world_size(sdp_group)
    sdp_rank = dist.get_rank(sdp_group)
    assert sdp_world_size == world_size
    assert 0 <= sdp_rank < sdp_world_size

    # Model parallel group (stub)
    assert ds_mpu.get_model_parallel_group() is None
    assert ds_mpu.get_model_parallel_world_size() == 1
    assert ds_mpu.get_model_parallel_rank() == 0


@pytest.mark.is_mpi
@pytest.mark.is_torch_gpu
def test_initialize_with_invalid_context_parallel_size():
    deepspeed.init_distributed()

    world_size = dist.get_world_size()
    invalid_context_parallel_size = world_size + 1

    with pytest.raises(ValueError, match="world_size.*divisible.*context_parallel_size"):
        ds_mpu.initialize(invalid_context_parallel_size)
