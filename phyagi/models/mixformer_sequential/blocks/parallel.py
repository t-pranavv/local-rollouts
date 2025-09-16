# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    SequenceParallel,
    parallelize_module,
)
from transformers import Cache, PretrainedConfig

from phyagi.models.mixformer_sequential.blocks.mixers import get_mixer
from phyagi.models.mixformer_sequential.blocks.mlps import get_mlp
from phyagi.models.mixformer_sequential.blocks.norms import (
    DEFAULT_NORM,
    TP_NORMS,
    get_norm,
)


class ParallelBlock(nn.Module):
    """Parallel block.

    This block applies parallel mixer and MLP layers to the input (used in GPT-J and CodeGen).

    """

    def __init__(
        self,
        config: PretrainedConfig,
        norm: Optional[Dict[str, Any]] = None,
        mixer: Optional[Dict[str, Any]] = None,
        mlp: Optional[Dict[str, Any]] = None,
        block_idx: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.block_idx = block_idx

        self.ln = get_norm(config.n_embd, norm_config=norm, eps=config.layer_norm_epsilon)
        self.mixer = get_mixer(config, mixer_config=mixer, mixer_idx=block_idx)
        self.mlp = get_mlp(config, mlp_config=mlp)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.norm_cls = norm.get("norm_cls", DEFAULT_NORM) if norm else DEFAULT_NORM
        self.gradient_checkpointing = config.gradient_checkpointing

    def set_torch_tp(self, tp_mesh: DeviceMesh, enable_sequence_parallel: bool = False, **kwargs) -> None:
        if tp_mesh.size() > 1:
            if self.norm_cls not in TP_NORMS:
                raise ValueError(
                    f"`norm_cls` must be one of {TP_NORMS} to use `set_torch_tp()`, but got '{self.norm_cls}'."
                )

            plan = {}

            if enable_sequence_parallel:
                plan.update({"ln": SequenceParallel()})
            if hasattr(self.mixer, "set_torch_tp"):
                plan.update(
                    {
                        "mixer": PrepareModuleInput(
                            input_layouts=(Shard(1),) if enable_sequence_parallel else (Replicate(),),
                            desired_input_layouts=(Replicate(),),
                        ),
                    }
                )
            if hasattr(self.mlp, "set_torch_tp"):
                plan.update(
                    {
                        "mlp": PrepareModuleInput(
                            input_layouts=(Shard(1),) if enable_sequence_parallel else (Replicate(),),
                            desired_input_layouts=(Replicate(),),
                        ),
                    }
                )

            parallelize_module(self, tp_mesh, plan)

    def set_torch_fsdp(self, dp_mesh: DeviceMesh, dp_config: Dict[str, Any], layer_id: int, n_layer: int) -> nn.Module:
        if dp_mesh.size() > 1:
            fully_shard(self, **dp_config, reshard_after_forward=layer_id < n_layer - 1)

    def forward(
        self,
        hidden_states: Union[torch.FloatTensor, Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        # `hidden_states` are wrapped as an extra tuple when using Pipeline Parallelism
        hidden_states = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        residual = hidden_states

        hidden_states = self.ln(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)

        attn_outputs = self.mixer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
        )
        attn_outputs = self.resid_dropout(attn_outputs)

        feed_forward_hidden_states = self.resid_dropout(feed_forward_hidden_states)
        hidden_states = attn_outputs + feed_forward_hidden_states + residual

        return (hidden_states,)
