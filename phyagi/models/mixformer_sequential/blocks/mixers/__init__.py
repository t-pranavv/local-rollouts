# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
from transformers import PretrainedConfig

from phyagi.models.mixformer_sequential.blocks.mixers.mha import MHA

MIXERS = {
    "mha": MHA,
}
DEFAULT_MIXER = "mha"


def get_mixer(
    config: PretrainedConfig, mixer_config: Optional[Dict[str, Any]] = None, mixer_idx: Optional[int] = None
) -> torch.nn.Module:
    mixer_config = copy.deepcopy(mixer_config)
    if mixer_config is not None:
        mixer_layer = mixer_config.pop("mixer_cls", None)
        if mixer_layer is None:
            raise ValueError("`mixer_cls` must be specified in `mixer_config`.")
        if mixer_layer not in MIXERS:
            raise ValueError(f"`mixer_cls` must be one of {list(MIXERS.keys())}, but got '{mixer_layer}'.")

        return MIXERS[mixer_layer](config, **mixer_config, layer_idx=mixer_idx)

    return MIXERS[DEFAULT_MIXER](config, layer_idx=mixer_idx)
