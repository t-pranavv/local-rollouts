# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field
from typing import Literal, Optional

from phyagi.rl.tuners.ray_worker_config import RayWorkerConfig


@dataclass
class RayGRPOConfig(RayWorkerConfig):
    """Ray GRPO configuration.

    Args:
        eval_group_size: Group size (completions per question) used during mid-training evaluation.
            If ``None``, defaults to ``group_size``.
        eval_batch_size: Batch size for evaluation.
            If ``None``, no batching is applied and all sequences are passed to vLLM.
        loss_normalization: Method to normalize the loss.
            ``batch`` normalizes by the number of valid tokens in the packed batch,
            ``sequence`` normalizes by the number of valid tokens in each sequence.
            ``max_context`` normalizes by the maximum context length accepted (``rollout.prompt_length`` + ``rollout.response_length``)
        num_policy_updates_per_batch: Number of policy updates per batch ('mu').
        kl_coeff: KL divergence term coefficient ('beta').
        entropy_coeff: Entropy coefficient for the policy loss. By default (``entropy_coeff=0.0``), disable entropy regularization.
        epsilon_low: Minimum epsilon clip value.
        epsilon_high: Maximum epsilon clip value.

    """

    eval_group_size: Optional[int] = field(
        default=None,
        metadata={"help": "Group size (completions per question) used during mid-training evaluation."},
    )

    eval_batch_size: Optional[int] = field(default=None, metadata={"help": "Batch size for evaluation."})

    loss_normalization: Literal["batch", "sequence", "max_context"] = field(
        default="batch",
        metadata={"help": "Method to normalize the loss."},
    )

    num_policy_updates_per_batch: int = field(
        default=1, metadata={"help": "Number of policy updates per batch ('mu')."}
    )

    kl_coeff: float = field(default=0.001, metadata={"help": "KL divergence term coefficient ('beta')."})

    entropy_coeff: float = field(
        default=0.0,
        metadata={"help": "Entropy coefficient for the policy loss. Set to 0.0 to disable entropy regularization."},
    )

    epsilon_low: float = field(default=0.2, metadata={"help": "Minimum epsilon clip value."})

    epsilon_high: float = field(default=0.2, metadata={"help": "Maximum epsilon clip value."})

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.num_policy_updates_per_batch > 1:
            raise NotImplementedError("`num_policy_updates_per_batch > 1` is not supported yet.")
