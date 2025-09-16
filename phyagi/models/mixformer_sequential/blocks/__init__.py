# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PretrainedConfig

from phyagi.models.mixformer_sequential.blocks.parallel import ParallelBlock
from phyagi.models.mixformer_sequential.blocks.sequential import SequentialBlock
from phyagi.utils.logging_utils import get_logger

logger = get_logger(__name__)


BLOCKS = {
    "parallel": ParallelBlock,
    "sequential": SequentialBlock,
}
DEFAULT_BLOCK = "parallel"


def get_block(
    config: PretrainedConfig, block_config: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
) -> torch.nn.Module:
    block_config = copy.deepcopy(block_config)
    if not isinstance(block_config, list):
        block_config = [block_config for _ in range(config.n_layer)]

    if config.n_layer != len(block_config):
        logger.warning(
            "`config.n_layer` does not match number of blocks in `block_config`. "
            + f"Overriding {config.n_layer} with {len(block_config)}."
        )
        config.n_layer = len(block_config)

    blocks = []
    for block_idx, block in enumerate(block_config):
        block = copy.deepcopy(block) or {"block_cls": DEFAULT_BLOCK}
        block_layer = block.pop("path", None) or block.pop("block_cls", None)

        if block_layer is None:
            raise ValueError("`block_cls` must be defined in `block_config`, but got None.")
        if block_layer not in BLOCKS:
            raise ValueError(f"`block_cls` must be one of {list(BLOCKS.keys())}, but got '{block_layer}'.")

        block["block_idx"] = block_idx
        blocks.append(BLOCKS[block_layer](config, **block))

    return blocks
