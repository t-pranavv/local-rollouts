# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple

import ray
import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


def _get_master_addr_port() -> Tuple[str, int]:
    master_addr = ray._private.services.get_node_ip_address()

    with socket.socket() as sock:
        sock.bind(("", 0))
        free_port = sock.getsockname()[1]

    return master_addr, free_port


@dataclass
class DistributedLayout:
    """Distributed parallelism configuration for the actor and rollout models.

    Supports tensor parallel (TP), context parallel (CP), and data parallel (DP)
    group construction and rank mapping across nodes and GPUs.

    Args:
        n_nodes: Number of nodes in the cluster.
        n_gpus_per_node: Number of GPUs per node (local world size).
        actor_cp_size: Context parallel size for the actor model.
        actor_tp_size: Tensor parallel size for the actor model.
        rollout_tp_size: Tensor parallel size for the rollout model.

    """

    n_nodes: int = field(metadata={"help": "Number of nodes in the cluster."})

    n_gpus_per_node: int = field(metadata={"help": "Number of GPUs per node (local world size)."})

    actor_cp_size: int = field(default=1, metadata={"help": "Context parallel size for the actor model."})

    actor_tp_size: int = field(default=1, metadata={"help": "Tensor parallel size for the actor model."})

    rollout_tp_size: int = field(default=1, metadata={"help": "Tensor parallel size for the rollout model."})

    world_size: int = field(init=False, metadata={"help": "Total number of processes in the distributed training."})

    rank_grid: Dict[str, torch.Tensor] = field(
        init=False, metadata={"help": "Grid of ranks for each context (actor and rollout)."}
    )

    @property
    def actor_dp_size(self) -> int:
        """Data parallel size for the actor model."""

        return self.world_size // (self.actor_tp_size * self.actor_cp_size)

    @property
    def rollout_dp_size(self) -> int:
        """Data parallel size for the rollout model."""

        return self.world_size // self.rollout_tp_size

    def __post_init__(self) -> None:
        self.world_size = self.n_nodes * self.n_gpus_per_node
        tp_sizes = {
            "actor": self.actor_tp_size,
            "rollout": self.rollout_tp_size,
        }

        # Tensor parallel and context parallel are only allowed intra-node
        for context, size in tp_sizes.items():
            if not (0 < size <= self.n_gpus_per_node):
                raise ValueError(f"`tp_size` for '{context}' must be 0 < {size} <= {self.n_gpus_per_node}.")
            if self.n_gpus_per_node % size != 0:
                raise ValueError(
                    f"`tp_size` for '{context}' must be divisible by `n_gpus_per_node`, but got {size} and {self.n_gpus_per_node}."
                )

        if not (0 < self.actor_cp_size <= self.n_gpus_per_node):
            raise ValueError(f"`cp_size` for 'actor' must be 0 < {self.actor_cp_size} <= {self.n_gpus_per_node}.")
        if self.n_gpus_per_node % self.actor_cp_size != 0:
            raise ValueError(
                f"`cp_size` for 'actor' must be divisible by `n_gpus_per_node`, but got {self.actor_cp_size} and {self.n_gpus_per_node}."
            )

        ranks = torch.arange(self.world_size)
        self.rank_grid = {
            "actor": ranks.reshape(-1, self.actor_cp_size, self.actor_tp_size),
            "rollout": ranks.reshape(-1, self.rollout_tp_size),
        }

    def _get_ranks(self, global_rank: int, context: Literal["actor", "rollout"]) -> Tuple[int, ...]:
        rank_grid = (self.rank_grid[context] == global_rank).nonzero()
        ranks = rank_grid.squeeze().tolist()

        expected_len = 3 if context == "actor" else 2
        if len(ranks) != expected_len:
            raise ValueError(f"Expected {expected_len} ranks for '{context}', but got {len(ranks)}.")

        return tuple(ranks)

    def init_device_mesh(self, context: Literal["actor", "rollout"]) -> DeviceMesh:
        """Initialize the device mesh for the given context (actor or rollout).

        Args:
            context: Context for which to initialize the device mesh. It can be either ``actor`` or ``rollout``.

        Returns:
            Initialized device mesh for the given context.

        """

        if context == "actor":
            shapes = (self.actor_dp_size, self.actor_cp_size, self.actor_tp_size)
            names = ["data_parallel", "context_parallel", "tensor_parallel"]
        elif context == "rollout":
            shapes = (self.rollout_dp_size, self.rollout_tp_size)
            names = ["data_parallel", "tensor_parallel"]

        device_mesh = init_device_mesh("cuda", mesh_shape=shapes, mesh_dim_names=names)
        if context == "actor":
            device_mesh[("data_parallel", "context_parallel")]._flatten(mesh_dim_name="data_context_parallel")

        return device_mesh

    def get_dp_rank(self, context: Literal["actor", "rollout"], global_rank: int) -> int:
        """Get the data parallel rank for the given context and global rank.

        Args:
            context: Context for which to get the data parallel rank.
            global_rank: Global rank of the process.

        Returns:
            Data parallel rank for the given context and global rank.

        """

        return self._get_ranks(global_rank, context)[0]

    def get_cp_rank(self, context: Literal["actor", "rollout"], global_rank: int) -> int:
        """Get the context parallel rank for the given context and global rank.

        Args:
            context: Context for which to get the context parallel rank.
            global_rank: Global rank of the process.

        Returns:
            Context parallel rank for the given context and global rank.

        """

        if context != "actor":
            raise ValueError("`get_cp_rank()` is only available for the actor.")

        return self._get_ranks(global_rank, context)[1]

    def get_tp_rank(self, context: Literal["actor", "rollout"], global_rank: int) -> int:
        """Get the tensor parallel rank for the given context and global rank.

        Args:
            context: Context for which to get the tensor parallel rank.
            global_rank: Global rank of the process.

        Returns:
            Tensor parallel rank for the given context and global rank.

        """

        return self._get_ranks(global_rank, context)[-1]

    def get_ray_worker_envs(self) -> List[Dict[str, str]]:
        """Get the environment variables for the Ray workers.

        Returns:
            List of dictionaries containing the environment variables for each worker.

        """

        master_addr, master_port = _get_master_addr_port()

        return [
            {
                "WORLD_SIZE": str(self.n_nodes * self.n_gpus_per_node),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
                "LOCAL_WORLD_SIZE": str(self.n_gpus_per_node),
                "LOCAL_RANK": str(j),
                "RANK": str(i * self.n_gpus_per_node + j),
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "NCCL_CUMEM_ENABLE": "0",
            }
            for i in range(self.n_nodes)
            for j in range(self.n_gpus_per_node)
        ]

    def distribute_data(self, data: List[Any], context: Literal["actor", "rollout"]) -> List[List[Any]]:
        """Distribute data across the workers for the given context (actor or rollout) respecting the distributed layout.

        Args:
            data: Data to be distributed. The datapoints are replicated across tensor parallel and
                context parallel groups, while different datapoints are distributed across data parallel groups.
                Each context parallel worker should manage splitting the data in the sequence dimension.
            context: Context for which to distribute the data.

        Returns:
            ``self.world_size`` sublists with the data for each worker.

        """

        dp_size = self.actor_dp_size if context == "actor" else self.rollout_dp_size
        indices = torch.arange(len(data)).tensor_split(dp_size)

        # Replicate the data across tensor parallel and context parallel groups
        return [
            [data[idx] for idx in indices[dp_group_idx]]
            for dp_group_idx, subgrid in enumerate(self.rank_grid[context])
            for _ in range(subgrid.numel())
        ]

    def gather_data(self, repeated_data: List[List[Any]], context: Literal["actor", "rollout"]) -> List[Any]:
        """Gather data from the workers for the given context (actor or rollout), deduplicating data
        that was replicated across tensor parallel and context parallel groups.

        Args:
            repeated_data: Data to be gathered, where the data is expected to be replicated
                across tensor parallel and context parallel groups.
            context: Context for which to gather the data.

        Returns:
            Gathered data with the original size.

        """

        dp_size = self.actor_dp_size if context == "actor" else self.rollout_dp_size

        # Deduplicate the data across tensor parallel and context parallel groups
        distinct_indices = range(0, self.world_size, self.world_size // dp_size)

        return [datapoint for index in distinct_indices for datapoint in repeated_data[index]]
