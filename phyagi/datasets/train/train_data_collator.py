# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import default_collate


@dataclass
class LMDataCollator:
    """Data collator compatible with :class:`phyagi.datasets.train.lm.lm_dataset.LMDataset` and
    :class:`phyagi.datasets.train.stream_lm.stream_lm_dataset.StreamLMDataset`.

    This class applies specific token masking based on given parameters.

    Args:
        ignore_token_ids: List of token IDs to be ignored in the label.
            Label tokens with these IDs will be replaced with the ``ignore_index`` value.
        ignore_token_id_range: Range of token IDs (inclusive) to be ignored (min, max) in the label.
            Any label token within this range will be replaced with the ``ignore_index`` value.
        ignore_index: The index that will be used to replace ignored tokens in the labels.

    Examples:
        >>> collator = LMDataCollator(ignore_token_ids=[0, 1], ignore_index=-1)
        >>> examples = [{'labels': torch.tensor([0, 1, 2, 3])}]
        >>> batch = collator(examples)
        >>> batch['labels']
        >>> tensor([-1, -1,  2,  3])

    """

    ignore_token_ids: Optional[List[int]] = field(default=None, metadata={"help": "List of token IDs to be ignored."})

    ignore_token_id_range: Optional[Tuple[int, int]] = field(
        default=None, metadata={"help": "Range of token IDs to be ignored (min, max)."}
    )

    ignore_index: int = field(default=-100, metadata={"help": "Index to be ignored in the labels."})

    def __post_init__(self) -> None:
        self.ignore_token_ids = torch.tensor(self.ignore_token_ids) if self.ignore_token_ids else None

    def _mask_tokens(self, labels: torch.LongTensor) -> torch.LongTensor:
        mask = torch.zeros_like(labels, dtype=torch.bool)

        if self.ignore_token_ids is not None:
            mask |= torch.isin(labels, self.ignore_token_ids)

        if self.ignore_token_id_range is not None:
            lower, upper = self.ignore_token_id_range

            lower = lower or float("-inf")
            upper = upper or float("inf")

            mask |= (labels >= lower) & (labels <= upper)

        labels[mask] = self.ignore_index

        return labels

    def __call__(self, examples: List[Dict[str, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        batch = default_collate(examples)

        if self.ignore_token_ids is not None or self.ignore_token_id_range is not None:
            if batch.get("labels", None) is not None:
                batch["labels"] = self._mask_tokens(batch["labels"])

        return batch
