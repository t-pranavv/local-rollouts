# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
from transformers import PretrainedConfig

from phyagi.models.mixformer_sequential.blocks.embeddings.common import (
    Embedding,
    PositionalEmbedding,
)

EMBEDDINGS = {
    "default": Embedding,
    "positional": PositionalEmbedding,
}
DEFAULT_EMBEDDING = "default"


def get_embedding(config: PretrainedConfig, embedding_config: Optional[Dict[str, Any]] = None) -> torch.nn.Module:
    embedding_config = copy.deepcopy(embedding_config)
    if embedding_config is not None:
        embedding_layer = embedding_config.pop("embedding_cls", None)
        if embedding_layer is None:
            raise ValueError("`embedding_cls` must be defined in `embedding_config`, but got None.")
        if embedding_layer not in EMBEDDINGS:
            raise ValueError(f"`embedding_cls` must be one of {list(EMBEDDINGS.keys())}, but got '{embedding_layer}'.")

        return EMBEDDINGS[embedding_layer](config, **embedding_config)

    return EMBEDDINGS[DEFAULT_EMBEDDING](config)
