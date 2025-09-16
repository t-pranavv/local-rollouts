# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import patch

import pytest

from phyagi.rl.distributed_layout import DistributedLayout


def generate_test_params():
    n_nodes_vals = [1, 2]
    n_gpus_vals = [1, 2, 4, 8]

    params = []
    for n_nodes in n_nodes_vals:
        for n_gpus in n_gpus_vals:
            for rollout_tp_size in range(1, n_gpus + 1):
                if n_gpus % rollout_tp_size == 0:
                    params.append((n_nodes, n_gpus, rollout_tp_size, 1, 1))

    for n_nodes in n_nodes_vals:
        for n_gpus in n_gpus_vals:
            for actor_tp_size in range(1, n_gpus + 1):
                for actor_cp_size in range(1, n_gpus - actor_tp_size + 1):
                    if (
                        n_gpus % actor_cp_size == 0
                        and n_gpus % actor_tp_size == 0
                        and actor_tp_size * actor_cp_size <= n_gpus
                    ):
                        params.append((n_nodes, n_gpus, 1, actor_tp_size, actor_cp_size))

    return list(set(params))


@pytest.mark.parametrize("n_nodes", [1, 3])
@pytest.mark.parametrize("n_gpus_per_node", [1, 8])
@patch("phyagi.rl.distributed_layout._get_master_addr_port", return_value=("127.0.0.1", 12345))
def test_get_dist_worker_envs(mock_get_addr_port, n_nodes, n_gpus_per_node):
    dist_layout = DistributedLayout(n_nodes=n_nodes, n_gpus_per_node=n_gpus_per_node)
    envs = dist_layout.get_ray_worker_envs()

    assert len(envs) == n_nodes * n_gpus_per_node
    assert envs[0]["WORLD_SIZE"] == str(n_nodes * n_gpus_per_node)
    assert envs[0]["MASTER_ADDR"] == mock_get_addr_port()[0]
    assert envs[0]["MASTER_PORT"] == str(mock_get_addr_port()[1])
    assert envs[0]["LOCAL_WORLD_SIZE"] == str(n_gpus_per_node)
    assert envs[0]["LOCAL_RANK"] == "0"
    assert envs[0]["RANK"] == "0"
    assert envs[-1]["RANK"] == str(n_nodes * n_gpus_per_node - 1)


@pytest.mark.parametrize(
    "n_nodes, n_gpus_per_node, rollout_tp_size, actor_tp_size, actor_cp_size", generate_test_params()
)
@pytest.mark.parametrize("length", [1, 16, 33])
def test_spread_gather_data(n_nodes, n_gpus_per_node, rollout_tp_size, actor_tp_size, actor_cp_size, length):
    dist_layout = DistributedLayout(n_nodes, n_gpus_per_node, rollout_tp_size, actor_tp_size, actor_cp_size)
    data = list(range(0, length))

    actor_spread_data = dist_layout.distribute_data(data, context="actor")
    rollout_spread_data = dist_layout.distribute_data(data, context="rollout")

    actor_spread_set = {element for batch in actor_spread_data for element in batch}
    rollout_spread_set = {element for batch in rollout_spread_data for element in batch}
    assert actor_spread_set == set(data)
    assert rollout_spread_set == set(data)

    gathered_actor_data = dist_layout.gather_data(actor_spread_data, "actor")
    gathered_rollout_data = dist_layout.gather_data(rollout_spread_data, "rollout")
    assert gathered_actor_data == data
    assert gathered_rollout_data == data
