# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import torch

from phyagi.models.mixformer_sequential.blocks.norms.low_precision import LPLayerNorm
from phyagi.models.mixformer_sequential.blocks.norms.rms import RMSLayerNorm
from phyagi.utils.import_utils import is_flash_attn_available

FlashRMSLayerNorm = RMSLayerNorm
if is_flash_attn_available():
    from flash_attn.ops.triton.layer_norm import RMSNorm as FlashRMSLayerNorm


NORMS = {
    "torch": torch.nn.LayerNorm,
    "low_precision": LPLayerNorm,
    "rms": RMSLayerNorm,
    "flash_rms": FlashRMSLayerNorm,
}
DEFAULT_NORM = "torch"
TP_NORMS = ["torch", "low_precision", "rms"]


def get_norm(shape: Tuple[int, ...], norm_config: Optional[Dict[str, Any]] = None, **kwargs) -> torch.nn.Module:
    norm_config = copy.deepcopy(norm_config)
    if norm_config is not None:
        norm_layer = norm_config.pop("norm_cls", None)
        if norm_layer is None:
            raise ValueError("`norm_cls` must be defined in `norm_config`, but got None.")

        if norm_layer not in NORMS:
            raise ValueError(f"`norm_cls` must be one of {list(NORMS.keys())}, but got '{norm_layer}'.")

        return NORMS[norm_layer](shape, **norm_config)

    return NORMS[DEFAULT_NORM](shape, **kwargs)
