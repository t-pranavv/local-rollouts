# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch


@dataclass
class PackedBatch:
    """Packed batch.

    Args:
        tokens: 2D tensor (rows x padded_length) of token identifiers.
        masks: 2D tensor (rows x padded_length) of boolean masks.
        advantages: 2D tensor (rows x padded_length) of advantages.
        cu_seqlens: 1D tensor of cumulative sequence lengths.
        valid_seqlens: 1D tensor of valid (mask=True) token lengths.
        position_ids: 2D tensor (rows x padded_length) of position identifiers.
        boundaries: Boundaries for each row in the packed batch.

    """

    tokens: torch.LongTensor = field(metadata={"help": "2D tensor of token identifiers."})

    masks: torch.BoolTensor = field(metadata={"help": "2D tensor of boolean masks."})

    advantages: torch.FloatTensor = field(metadata={"help": "2D tensor of advantages."})

    cu_seqlens: torch.IntTensor = field(metadata={"help": "1D tensor of cumulative sequence lengths."})

    valid_seqlens: torch.IntTensor = field(metadata={"help": "1D tensor of valid (mask=True) token lengths."})

    position_ids: Optional[torch.LongTensor] = field(
        default=None, metadata={"help": "2D tensor of position identifiers."}
    )

    boundaries: Optional[List[List[int]]] = field(
        default=None,
        metadata={"help": "Boundaries for each row in the packed batch."},
    )

    def to(self, device: torch.device) -> PackedBatch:
        """Move the packed batch to the specified device.

        Args:
            device: Target device (e.g., 'cuda' or 'cpu').

        Returns:
            Instance with tensors moved to the specified device.

        """

        self.tokens = self.tokens.to(device)
        self.masks = self.masks.to(device)
        self.advantages = self.advantages.to(device)
        self.cu_seqlens = self.cu_seqlens.to(device)
        self.valid_seqlens = self.valid_seqlens.to(device)
        if self.position_ids is not None:
            self.position_ids = self.position_ids.to(device)

        return self

    def to_dict(self) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """Convert the packed batch to a dictionary of tensors.

        Returns:
            Dictionary with attributes as keys and tensors as values.

        """

        return {
            "tokens": self.tokens,
            "masks": self.masks,
            "advantages": self.advantages,
            "cu_seqlens": self.cu_seqlens,
            "valid_seqlens": self.valid_seqlens,
            "position_ids": self.position_ids,
            "boundaries": self.boundaries,
        }

    @classmethod
    def from_dict(cls: PackedBatch, data: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> PackedBatch:
        """Create a packed batch from a dictionary of tensors.

        Args:
            data: Dictionary with attributes as keys and tensors as values.

        Returns:
            Instance with tensors set from the dictionary.

        """

        return cls(
            tokens=data["tokens"],
            masks=data["masks"],
            advantages=data["advantages"],
            cu_seqlens=data["cu_seqlens"],
            valid_seqlens=data["valid_seqlens"],
            position_ids=data.get("position_ids", None),
            boundaries=data.get("boundaries", None),
        )


def _build_padded_tensor_and_flattened_cu_seqlens(
    bucket: List[Dict[str, List[tuple]]],
    pad_token_id: int,
    max_seq_len: int,
) -> PackedBatch:
    tokens_list = []
    masks_list = []
    advs_list = []
    per_row_boundaries = []

    for row in bucket:
        row_data = row["data"]
        row_boundaries = row["boundaries"].copy()

        if row_data:
            tokens, masks, advs = zip(*row_data)
            tokens, masks, advs = list(tokens), list(masks), list(advs)
        else:
            tokens, masks, advs = [], [], []

        pad_length = max_seq_len - len(tokens)
        tokens.extend([pad_token_id] * pad_length)
        masks.extend([False] * pad_length)
        advs.extend([0.0] * pad_length)

        tokens_list.append(tokens)
        masks_list.append(masks)
        advs_list.append(advs)

        # Ensure boundaries include the padded region
        if row_boundaries[-1] < max_seq_len:
            row_boundaries.append(max_seq_len)
        per_row_boundaries.append(row_boundaries)

    tokens_tensor = torch.tensor(tokens_list, dtype=torch.long)
    masks_tensor = torch.tensor(masks_list, dtype=torch.bool)
    advs_tensor = torch.tensor(advs_list, dtype=torch.float)

    # Compute flattened cumulative sequence lengths
    flattened_cu = []
    offset = 0
    for i, boundaries in enumerate(per_row_boundaries):
        if i == 0:
            flattened_cu.extend([b + offset for b in boundaries])
        else:
            # Skip the duplicate boundary at the start of each row
            flattened_cu.extend([b + offset for b in boundaries[1:]])
        offset += max_seq_len
    cu_seqlens_tensor = torch.tensor(flattened_cu, dtype=torch.int32)

    # Compute number of valid tokens per sequence
    valid_seqlens = []
    for row_idx, boundaries in enumerate(per_row_boundaries):
        row_masks = masks_tensor[row_idx]
        for j in range(len(boundaries) - 1):
            start_idx, end_idx = boundaries[j], boundaries[j + 1]
            valid_seqlens.append(torch.sum(row_masks[start_idx:end_idx]).item())
    valid_seqlens_tensor = torch.tensor(valid_seqlens, dtype=torch.int32)

    return PackedBatch(
        tokens=tokens_tensor,
        masks=masks_tensor,
        advantages=advs_tensor,
        cu_seqlens=cu_seqlens_tensor,
        valid_seqlens=valid_seqlens_tensor,
        boundaries=per_row_boundaries,
    )


def distribute_and_pack_sequences(
    sequences: List[List[int]],
    assistant_masks: List[List[bool]],
    advantages: List[float],
    max_length: int,
    dp_size: int = 1,
    cp_size: int = 1,
    tp_size: int = 1,
    pad_token_id: int = -1,
    normalize_adv_num_tokens: bool = False,
    micro_batch_size: Optional[int] = None,
    pad_to_largest_micro_batch: bool = False,
) -> Dict[int, List[PackedBatch]]:
    """Pack token sequences into GPU buckets and (if requested) splits each bucket into micro batches.

    The input sequences (with matching ``assistant_masks`` and ``advantages``) are first processed by pairing
    each token with its corresponding mask and advantage (normalized if needed). The sequences are then
    packed into GPU buckets (each bucket is a list of rows containing one or more packed sequences).

    If ``micro_batch_size`` is provided, each GPU bucket's rows are split into micro batches (each micro batch
    is a sublist of rows of ``length <= micro_batch_size``). For each micro batch, padded tensors and a
    flattened cumulative sequence length vector (``cu_seqlens``) are created.

    If ``pad_to_largest_micro_batch=True``, the micro batches are padded so each device receives the same number of
    micro batches. This is useful for distributed training to avoid gradient synchronization issues with FSDP.
    The extra micro batches added will contain just 2 padding tokens.

    Args:
        sequences: List of sequences (each sequence is a list of token identifiers).
        assistant_masks: List of boolean masks indicating which tokens are valid.
        advantages: List of advantage values for each token.
        max_length: Maximum length of sequences (used for padding).
        dp_size: Number of data parallel workers.
        cp_size: Number of context parallel workers.
        tp_size: Number of tensor parallel workers.
        pad_token_id: Token identifier used for padding.
        normalize_adv_num_tokens: Whether to normalize the advantage by the number of valid tokens.
        micro_batch_size: Size of micro batches.
            If ``None``, no micro batching is applied.
        pad_to_largest_micro_batch: Whether to pad micro batches to the largest size across GPUs.

    Returns:
      Dictionary mapping GPU index to a list of ``PackedBatch`` instances (one per micro batch). Even if
      micro batching is not applied, the GPU index will map to a list containing a single ``PackedBatch``.

    """

    if not all(len(seq) <= max_length for seq in sequences):
        raise ValueError(
            f"`sequences` must not exceed `max_length`, but got {[len(seq) for seq in sequences]} and {max_length}."
        )
    if not (len(sequences) == len(assistant_masks) == len(advantages)):
        raise ValueError(
            f"`sequences`, `assistant_masks`, and `advantages` must have the same length, but got {len(sequences)}, {len(assistant_masks)}, and {len(advantages)}."
        )

    processed_seqs = []
    for seq, mask_seq, adv in zip(sequences, assistant_masks, advantages):
        if isinstance(seq, np.ndarray):
            seq = seq.tolist()

        n_valid = sum(mask_seq)
        adv = adv / n_valid if (normalize_adv_num_tokens and n_valid) else adv

        processed_seqs.append([(token, mask, adv if mask else 0.0) for token, mask in zip(seq, mask_seq)])

    sorted_idxs = sorted(range(len(processed_seqs)), key=lambda i: len(processed_seqs[i]), reverse=True)

    def _pack_for_one_worker(idx_subset: List[int]) -> List["PackedBatch"]:
        bucket = {"rows": [{"data": [], "boundaries": [0]}], "current_length": 0}
        remaining = idx_subset.copy()

        while remaining:
            row = bucket["rows"][-1]
            added = False
            for i in remaining:
                seq = processed_seqs[i]
                if bucket["current_length"] + len(seq) <= max_length:
                    row["data"].extend(seq)
                    row["boundaries"].append(row["boundaries"][-1] + len(seq))
                    bucket["current_length"] += len(seq)
                    remaining.remove(i)
                    added = True
                    break
            if not added:
                i = remaining.pop(0)
                seq = processed_seqs[i]
                bucket["rows"].append({"data": seq.copy(), "boundaries": [0, len(seq)]})
                bucket["current_length"] = len(seq)

        rows = bucket["rows"]

        if micro_batch_size is None:
            micro_batches = [rows]
        else:
            micro_batches = [rows[i : i + micro_batch_size] for i in range(0, len(rows), micro_batch_size)]

        return [_build_padded_tensor_and_flattened_cu_seqlens(mb, pad_token_id, max_length) for mb in micro_batches]

    dp_chunks = [sorted_idxs[i::dp_size] for i in range(dp_size)]
    dp_packs = [_pack_for_one_worker(chunk) for chunk in dp_chunks]

    if pad_to_largest_micro_batch:
        global_max_mb = max(len(packs) for packs in dp_packs)

        dummy_row = {"data": [(pad_token_id, False, 0.0)] * 2, "boundaries": [0, 2]}
        dummy_batch = _build_padded_tensor_and_flattened_cu_seqlens([dummy_row], pad_token_id, max_length)

        for packs in dp_packs:
            while len(packs) < global_max_mb:
                packs.append(copy.deepcopy(dummy_batch))

    gpu_batches = {}
    for dp_rank, packs in enumerate(dp_packs):
        for cp_rank in range(cp_size):
            for tp_rank in range(tp_size):
                global_rank = (dp_rank * cp_size + cp_rank) * tp_size + tp_rank
                gpu_batches[global_rank] = copy.deepcopy(packs)

    return gpu_batches
