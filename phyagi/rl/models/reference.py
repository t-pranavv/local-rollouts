# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from copy import deepcopy
from logging import Logger

import torch
from torch.distributed.device_mesh import DeviceMesh

from phyagi.models.mixformer_sequential.parallel_mixformer_sequential import (
    apply_cp_mixformer_sequential,
    apply_fsdp_mixformer_sequential,
    apply_tp_mixformer_sequential,
)
from phyagi.rl.models.actor import Actor, ActorConfig
from phyagi.utils.checkpoint import CheckpointManager


class Reference(Actor):
    """Ray reference model (with FSDP support)."""

    def __init__(
        self, config: ActorConfig, device_mesh: DeviceMesh, checkpoint_manager: CheckpointManager, logger: Logger
    ) -> None:
        """Initialize the reference model.

        Args:
            config: Reference model configuration.
            device_mesh: Device mesh.
            checkpoint_manager: Checkpoint manager for saving/loading checkpoints.
            logger: Logger for the reference model.

        """

        self.config = deepcopy(config)
        self.device_mesh = device_mesh
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger

        self.manual_offload = False
        self.fsdp_offload = True
        self.precision = self.config.dtype

        # TP requires a full state dict to avoid memory pressure when using asynchronous checkpointing
        self.full_state_dict = self.checkpoint_manager.mode == "async" and self.config.tensor_parallel_size > 1

        self.model = self.configure_model(**self.config.model)

        cp_mesh = self.device_mesh["context_parallel"]
        tp_mesh = self.device_mesh["tensor_parallel"]
        dp_mesh = self.device_mesh["data_context_parallel"]

        if cp_mesh.size() > 1:
            apply_cp_mixformer_sequential(self.model, cp_mesh, varlen=True)
        if tp_mesh.size() > 1:
            apply_tp_mixformer_sequential(self.model, tp_mesh, enable_loss_parallel=self.config.tp_loss_parallel)
        apply_fsdp_mixformer_sequential(self.model, dp_mesh, self.precision, cpu_offload=self.fsdp_offload)

        # Since `fsdp_offload` is enabled, we need to ensure that there is no GPU cache allocated
        torch.cuda.empty_cache()

        # Update the model configuration with the full configuration (with defaults and user overrides)
        # and set the precision to the one specified in the config
        self.config.model = self.model.config.to_diff_dict()
        self.config.model["torch_dtype"] = self.precision
