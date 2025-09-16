# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from transformers import PretrainedConfig

from phyagi.models.mixformer_sequential.blocks.norms import get_norm
from phyagi.utils.import_utils import is_flash_attn_available
from phyagi.utils.logging_utils import get_logger
from phyagi.utils.type_utils import to_torch_dtype

FlashCrossEntropyLoss = None
if is_flash_attn_available():
    from flash_attn.losses.cross_entropy import (
        CrossEntropyLoss as FlashCrossEntropyLoss,
    )

logger = get_logger(__name__)


class CausalLMHead(nn.Module):
    """Causal Language Modeling head."""

    def __init__(self, config: PretrainedConfig, dtype: Optional[str] = None, use_bias: bool = True) -> None:
        super().__init__()

        self.norm_config = config.architecture.get("norm", None) if isinstance(config.architecture, dict) else None
        self.dtype = to_torch_dtype(dtype)

        self.ln = get_norm(config.n_embd, norm_config=self.norm_config, eps=config.layer_norm_epsilon, dtype=self.dtype)
        self.linear = nn.Linear(config.n_embd, config.vocab_size, bias=use_bias, dtype=self.dtype)

    def set_torch_tp(
        self, tp_mesh: DeviceMesh, enable_sequence_parallel: bool = False, enable_loss_parallel: bool = False, **kwargs
    ) -> None:
        if tp_mesh.size() > 1:
            plan = {
                "linear": ColwiseParallel(
                    input_layouts=Shard(1) if enable_sequence_parallel else Replicate(),
                    output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                    use_local_output=not enable_loss_parallel,
                ),
            }
            if enable_sequence_parallel:
                plan.update({"ln": SequenceParallel()})

            parallelize_module(self, tp_mesh, plan)

    def forward(self, hidden_states: Union[torch.FloatTensor, Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        # DeepSpeed's pipeline parallelism wraps inputs with an extra tuple
        hidden_states = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states

        hidden_states = self.ln(hidden_states)
        logits = self.linear(hidden_states)

        return logits


class CausalLMLoss(nn.Module):
    """Causal Language Modeling loss."""

    def __init__(
        self,
        shift_labels: bool = True,
        ignore_index: int = -100,
        reduction: str = "mean",
        flash_cross_entropy_loss: bool = False,
        inplace_backward: bool = False,
    ) -> None:
        super().__init__()

        self.shift_labels = shift_labels
        self.loss_fct = (
            FlashCrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, inplace_backward=inplace_backward)
            if flash_cross_entropy_loss and FlashCrossEntropyLoss is not None
            else nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        )

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        if self.shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        return self.loss_fct(logits.to(torch.float32).view(-1, logits.size(-1)), labels.view(-1))
