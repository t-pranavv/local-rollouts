# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module
from transformers import PretrainedConfig
from transformers.activations import ACT2FN

from phyagi.models.parallel_utils import ShardColwiseParallel
from phyagi.utils.import_utils import is_flash_attn_available
from phyagi.utils.logging_utils import get_logger
from phyagi.utils.type_utils import to_torch_dtype

swiglu = None
if is_flash_attn_available():
    from flash_attn.ops.activations import swiglu


logger = get_logger(__name__)


class GLU(nn.Module):
    """Gated Linear Unit."""

    def __init__(
        self,
        config: PretrainedConfig,
        n_inner: Optional[int] = None,
        act_fn: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.act_fn = config.activation_function if act_fn is None else act_fn
        if self.act_fn not in ACT2FN.keys():
            raise ValueError(f"`act_fn` must be one of {list(ACT2FN.keys())}, but got '{act_fn}'.")
        self.act = ACT2FN[self.act_fn]

        n_inner = getattr(config, "n_inner", None) if n_inner is None else n_inner
        self.n_inner = n_inner if n_inner is not None else 4 * config.n_embd

        self.fc1 = nn.Linear(config.n_embd, 2 * self.n_inner, bias=False, dtype=to_torch_dtype(dtype))
        self.fc2 = nn.Linear(self.n_inner, config.n_embd, bias=False, dtype=to_torch_dtype(dtype))

        self.tp_mesh = None
        self.is_torch_tp = False

        self.register_state_dict_post_hook(self._unshard_state_dict_hook)

    @staticmethod
    def _unshard_state_dict_hook(module, state_dict, prefix, local_metadata):
        key = prefix + "fc1.weight"
        if key not in state_dict:
            return

        # Skip if the module does not have tensor parallelism enabled
        if module.tp_mesh is None:
            return

        shard_sizes = [module.n_inner, module.n_inner]
        state_dict[key] = ShardColwiseParallel.unshard_param(state_dict[key], shard_sizes, module.tp_mesh)

    def set_torch_tp(self, tp_mesh: DeviceMesh, enable_sequence_parallel: bool = False, **kwargs) -> None:
        if tp_mesh.size() > 1:
            plan = {
                "fc1": ShardColwiseParallel(shard_sizes=[self.n_inner, self.n_inner]),
                "fc2": RowwiseParallel(output_layouts=Shard(1) if enable_sequence_parallel else Replicate()),
            }

            parallelize_module(self, tp_mesh, plan)

            self.tp_mesh = tp_mesh
            self.is_torch_tp = True

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        y = self.fc1(hidden_states)

        y, gate = y.chunk(2, dim=-1)
        if self.act_fn == "silu" and not self.is_torch_tp and swiglu is not None:
            y = swiglu(gate, y)
        else:
            y = y * self.act(gate)

        return self.fc2(y)
