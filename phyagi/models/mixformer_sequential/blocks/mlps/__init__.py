# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
from transformers import PretrainedConfig

from phyagi.models.mixformer_sequential.blocks.mlps.glu import GLU
from phyagi.models.mixformer_sequential.blocks.mlps.mlp import MLP, FusedMLP

MLPS = {
    "glu": GLU,
    "fused_mlp": FusedMLP,
    "mlp": MLP,
}
DEFAULT_MLP = "fused_mlp"


def get_mlp(config: PretrainedConfig, mlp_config: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
    mlp_config = copy.deepcopy(mlp_config)
    if mlp_config is not None:
        mlp_layer = mlp_config.pop("mlp_cls", None)
        if mlp_layer is None:
            raise ValueError("`mlp_cls` must be defined in `mlp_config`, but got None.")

        if mlp_layer not in MLPS:
            raise ValueError(f"`mlp_cls` must be one of {list(MLPS.keys())}, but got '{mlp_layer}'.")

        return MLPS[mlp_layer](config, **mlp_config)

    return MLPS[DEFAULT_MLP](config)
