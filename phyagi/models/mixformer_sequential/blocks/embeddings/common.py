# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module
from transformers import PretrainedConfig


class Embedding(nn.Module):
    """Token embedding with dropout."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

    def set_torch_tp(self, tp_mesh: DeviceMesh, enable_sequence_parallel: bool = False, **kwargs) -> None:
        if tp_mesh.size() > 1:
            plan = {
                "wte": RowwiseParallel(
                    input_layouts=Replicate(), output_layouts=Shard(1) if enable_sequence_parallel else Replicate()
                ),
            }
            parallelize_module(self, tp_mesh, plan)

    def forward(self, input_ids: torch.LongTensor, **kwargs) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.wte(input_ids)
        hidden_states = self.drop(hidden_states)

        return hidden_states


class PositionalEmbedding(nn.Module):
    """Token embedding with positional encoding."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(
        self, input_ids: torch.LongTensor, position_ids: Optional[torch.LongTensor] = None, **kwargs
    ) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        input_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        hidden_states = input_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        return hidden_states
