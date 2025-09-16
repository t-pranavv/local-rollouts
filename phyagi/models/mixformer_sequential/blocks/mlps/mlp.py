# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import PretrainedConfig
from transformers.activations import ACT2FN

from phyagi.utils.import_utils import is_flash_attn_available
from phyagi.utils.logging_utils import get_logger
from phyagi.utils.type_utils import to_torch_dtype

FlashAttnFusedMLP = None
if is_flash_attn_available():
    from flash_attn.modules.mlp import FusedMLP as FlashAttnFusedMLP


logger = get_logger(__name__)


class MLP(nn.Module):
    """Multi-Layer Perceptron."""

    def __init__(
        self,
        config: PretrainedConfig,
        n_inner: Optional[int] = None,
        act_fn: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        act_fn = config.activation_function if act_fn is None else act_fn
        if act_fn not in ACT2FN.keys():
            raise ValueError(f"`act_fn` must be one of {list(ACT2FN.keys())}, but got '{act_fn}'.")
        self.act = ACT2FN[act_fn]

        n_inner = getattr(config, "n_inner", None) if n_inner is None else n_inner
        self.n_inner = n_inner if n_inner is not None else 4 * config.n_embd

        self.fc1 = nn.Linear(config.n_embd, self.n_inner, dtype=to_torch_dtype(dtype))
        self.fc2 = nn.Linear(self.n_inner, config.n_embd, dtype=to_torch_dtype(dtype))

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        old_keys = [prefix + "fc_in.weight", prefix + "fc_out.weight", prefix + "fc_in.bias", prefix + "fc_out.bias"]
        new_keys = [prefix + "fc1.weight", prefix + "fc2.weight", prefix + "fc1.bias", prefix + "fc2.bias"]

        if all(k in state_dict for k in old_keys) and not all(k in state_dict for k in new_keys):
            # Older version of `MLP` saved with different key names.
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def set_torch_tp(self, tp_mesh: DeviceMesh, enable_sequence_parallel: bool = False, **kwargs) -> None:
        if tp_mesh.size() > 1:
            plan = {
                "fc1": ColwiseParallel(),
                "fc2": RowwiseParallel(output_layouts=Shard(1) if enable_sequence_parallel else Replicate()),
            }
            parallelize_module(self, tp_mesh, plan)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class FusedMLP(nn.Module):
    """Fused Multi-Layer Perceptron."""

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: Optional[str] = None,
        n_inner: Optional[int] = None,
        act_fn: Optional[str] = None,
    ) -> None:
        super().__init__()

        act_fn = config.activation_function if act_fn is None else act_fn
        if act_fn not in ACT2FN.keys():
            raise ValueError(f"`act_fn` must be one of {list(ACT2FN.keys())}, but got '{act_fn}'.")
        gelu_activations = ["gelu_new", "gelu_fast", "gelu_approx"]

        n_inner = getattr(config, "n_inner", None) if n_inner is None else n_inner
        n_inner = n_inner if n_inner is not None else 4 * config.n_embd

        self.mlp = (
            MLP(config, dtype=dtype, n_inner=n_inner, act_fn=act_fn)
            if FlashAttnFusedMLP is None
            else FlashAttnFusedMLP(
                in_features=config.n_embd,
                hidden_features=n_inner,
                activation="gelu_approx" if act_fn in gelu_activations else "relu",
                dtype=to_torch_dtype(dtype),
            )
        )

    def set_torch_tp(self, tp_mesh: DeviceMesh, **kwargs) -> None:
        if tp_mesh.size() > 1:
            raise ValueError("`FusedMLP` is not supported with `set_torch_tp()`.")

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return self.mlp(hidden_states)
