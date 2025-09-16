# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Tuple, Union

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from transformers import PretrainedConfig


class SequenceClassificationHead(nn.Module):
    """Sequence classification head."""

    def __init__(self, config: PretrainedConfig, use_bias: bool = False) -> None:
        super().__init__()

        self.num_labels = config.num_labels
        self.linear = nn.Linear(config.n_embd, self.num_labels, bias=use_bias)

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
            parallelize_module(self, tp_mesh, plan)

    def forward(self, hidden_states: Union[torch.FloatTensor, Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        # DeepSpeed's pipeline parallelism wraps inputs with an extra tuple
        hidden_states = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states

        logits = self.linear(hidden_states)

        return logits


class SequenceClassificationLoss(nn.Module):
    """Sequence classification loss."""

    _LOSS_FUNCTIONS = {
        "regression": nn.MSELoss(),
        "single_label_classification": nn.CrossEntropyLoss(),
        "multi_label_classification": nn.BCEWithLogitsLoss(),
    }

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()

        if config.problem_type not in self._LOSS_FUNCTIONS:
            raise ValueError(
                f"`problem_type` must be one of {list(self._LOSS_FUNCTIONS.keys())}, but got '{config.problem_type}'."
            )

        self.problem_type = config.problem_type
        self.num_labels = config.num_labels
        self.loss_fct = self._LOSS_FUNCTIONS[self.problem_type]

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        if self.problem_type == "regression":
            if self.num_labels == 1:
                return self.loss_fct(logits.squeeze(), labels.squeeze())
            return self.loss_fct(logits, labels)

        if self.problem_type == "single_label_classification":
            return self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return self.loss_fct(logits, labels)
