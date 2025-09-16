# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
from transformers import PretrainedConfig

from phyagi.models.mixformer_sequential.blocks.heads.causal_lm import (
    CausalLMHead,
    CausalLMLoss,
)
from phyagi.models.mixformer_sequential.blocks.heads.seq_cls import (
    SequenceClassificationHead,
    SequenceClassificationLoss,
)

HEADS = {
    "causal_lm": CausalLMHead,
    "seq_cls": SequenceClassificationHead,
}
DEFAULT_HEAD = "causal_lm"

LOSSES = {
    "causal_lm": CausalLMLoss,
    "seq_cls": SequenceClassificationLoss,
}
DEFAULT_LOSS = "causal_lm"


def get_head(config: PretrainedConfig, head_config: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
    head_config = copy.deepcopy(head_config)
    if head_config is not None:
        head_layer = head_config.pop("head_cls", None)
        if head_layer is None:
            raise ValueError("`head_cls` must be defined in `head_config`, but got None.")
        if head_layer not in HEADS:
            raise ValueError(f"`head_cls` must be one of {list(HEADS.keys())}, but got '{head_layer}'.")

        return HEADS[head_layer](config, **head_config)

    return HEADS[DEFAULT_HEAD](config)


def get_loss(*args, loss_config: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
    loss_config = copy.deepcopy(loss_config)
    if loss_config is not None:
        loss_layer = loss_config.pop("loss_cls", None)
        if loss_layer is None:
            raise ValueError("`loss_cls` must be defined in `loss_config`, but got None.")
        if loss_layer not in LOSSES:
            raise ValueError(f"`loss_cls` must be one of {list(LOSSES.keys())}, but got '{loss_layer}'.")

        return LOSSES[loss_layer](*args, **loss_config)

    return LOSSES[DEFAULT_LOSS](*args)
