# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import ModelParallelStrategy
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from typing_extensions import override


def _setup_device_mesh(
    data_parallel_size: int,
    context_parallel_size: int,
    tensor_parallel_size: int,
    world_size: int,
    device: torch.device,
) -> DeviceMesh:
    if data_parallel_size * context_parallel_size * tensor_parallel_size != world_size:
        raise ValueError(
            f"`data_parallel_size * context_parallel_size * tensor_parallel_size` must be {world_size}, but got {data_parallel_size * context_parallel_size * tensor_parallel_size}."
        )

    device_mesh = init_device_mesh(
        device_type=device.type,
        mesh_shape=(data_parallel_size, context_parallel_size, tensor_parallel_size),
        mesh_dim_names=("data_parallel", "context_parallel", "tensor_parallel"),
    )

    # Set additional communication subgroups for the device mesh
    device_mesh[("data_parallel", "context_parallel")]._flatten(mesh_dim_name="data_context_parallel")

    return device_mesh


class DataContextTensorParallelStrategy(ModelParallelStrategy):
    """Parallel strategy that supports up to 3D-parallelism (data with FSDP, context, and tensor)."""

    STRATEGY_TYPE = "dctp"

    def __init__(
        self,
        data_parallel_size: int = 1,
        context_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        cpu_offload: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the parallel strategy.

        Args:
            data_parallel_size: Number of data parallel processes.
            context_parallel_size: Number of context parallel processes.
            tensor_parallel_size: Number of tensor parallel processes.
            cpu_offload: Whether to offload the CPU.

        """

        # When using FSDP with CPU offload, we need to use GLOO as the CPU process group backend
        # https://github.com/pytorch/torchtune/pull/2108
        if cpu_offload:
            kwargs["process_group_backend"] = "cuda:nccl,cpu:gloo"

        super().__init__(data_parallel_size=data_parallel_size, tensor_parallel_size=tensor_parallel_size, **kwargs)

        self._cpu_offload = cpu_offload
        self._context_parallel_size = context_parallel_size

    def setup(self, trainer: Trainer) -> None:
        super().setup(trainer)

        # When using `cpu_offload`, we need to move the model to the CPU
        # https://github.com/pytorch/torchtune/issues/1977
        if self._cpu_offload:
            trainer.lightning_module.model.to("cpu")

    @override
    def setup_environment(self) -> None:
        if self.accelerator is None:
            raise ValueError("`accelerator` must be defined, but got None.")
        self.accelerator.setup_device(self.root_device)

        self._setup_distributed()

        self._device_mesh = _setup_device_mesh(
            self._data_parallel_size,
            self._context_parallel_size,
            self._tensor_parallel_size,
            self.world_size,
            self.root_device,
        )

        if self.lightning_module is None:
            raise ValueError("`lightning_module` must be defined, but got None.")
        self.lightning_module._device_mesh = self._device_mesh
