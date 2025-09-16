# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RewardDataCollator:
    """Data collator compatible with :class:`torch.utils.data.Dataset`.

    Args:
        reward_names: List of reward names.
        reward_weights: List of reward weights.
            If ``None``, equal weights are used.

    Examples:
        >>> collator = RewardDataCollator(reward_names=["reward1", "reward2"], reward_weights=[0.5, 0.5])
        >>> examples = [{'input_ids': torch.tensor([0, 1, 2, 3])}]
        >>> batch = collator(examples)
        >>> batch
        >>> {'reward_names': [['reward1', 'reward2']], 'reward_weights': [[0.5, 0.5]], 'input_ids': [tensor([0, 1, 2, 3])]}

    """

    reward_names: List[str] = field(default_factory=list, metadata={"help": "List of reward names."})

    reward_weights: Optional[List[float]] = field(default=None, metadata={"help": "List of reward weights."})

    def __post_init__(self) -> None:
        if len(self.reward_names) == 0:
            raise ValueError("`reward_names` must have at least one item, but got 0.")
        if self.reward_weights is not None and (len(self.reward_weights) != len(self.reward_names)):
            raise ValueError(
                f"`reward_weights` and `reward_names` must have the same length, but got {len(self.reward_weights)} and {len(self.reward_names)}."
            )

        self.reward_weights = self.reward_weights or [1.0 / len(self.reward_names)] * len(self.reward_names)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            "reward_names": [self.reward_names] * len(examples),
            "reward_weights": [self.reward_weights] * len(examples),
        }

        for example in examples:
            for k, v in example.items():
                if k in ["reward_names", "reward_weights"]:
                    continue
                if k not in batch:
                    batch[k] = []
                batch[k].append(v)

        return batch
