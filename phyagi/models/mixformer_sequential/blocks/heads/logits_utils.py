# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from phyagi.utils.import_utils import is_flash_attn_available

cross_entropy_loss = None
if is_flash_attn_available():
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss


def logprobs_from_parallel_logits(logits: DTensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes log probabilities from vocabulary-parallel logits.

    This function takes logits that are sharded across the vocabulary dimension (tensor parallel)
    and computes the log probabilities for the given labels.

    Args:
        logits: Logits distributed across tensor parallel workers.
        labels: Labels for which to compute log probabilities.

    Returns:
        Log probabilities for the given labels.

    """

    tp_mesh = logits.device_mesh
    tp_pg = tp_mesh.get_group()

    local_logits = logits.to_local()
    batch_size, seq_len, vocab_size = local_logits.shape

    flat_logits = local_logits.reshape(-1, vocab_size)
    flat_labels = labels.roll(-1, -1).reshape(-1)

    loss, _ = cross_entropy_loss(
        flat_logits,
        flat_labels,
        process_group=tp_pg,
    )

    logprobs = (-loss).view(batch_size, seq_len)[:, :-1]
    return logprobs.contiguous()


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes log probabilities from logits.

    This function computes the log probabilities for the given labels based on the provided logits.

    Args:
        logits: Logits.
        labels: Labels for which to compute log probabilities.

    Returns:
        Log probabilities for the given labels.

    """

    logits, labels = logits[:, :-1], labels[:, 1:]
    batch_size, seq_len, vocab_size = logits.shape

    if cross_entropy_loss is not None:
        flat_logits = logits.reshape(-1, vocab_size)
        flat_labels = labels.reshape(-1)

        loss, _ = cross_entropy_loss(
            flat_logits,
            flat_labels,
        )
        return (-loss).view(batch_size, seq_len)

    if logits.dtype in (torch.float32, torch.float64):
        logits_labels = torch.gather(logits, -1, labels.unsqueeze(-1)).squeeze(-1)
        logsumexp = torch.logsumexp(logits, dim=-1)
        return logits_labels - logsumexp

    rows = []
    for row_logits, row_labels in zip(logits, labels):
        logprobs = F.log_softmax(row_logits, dim=-1)
        rows.append(logprobs.gather(-1, row_labels.unsqueeze(-1)).squeeze(-1))

    return torch.stack(rows)


def entropy_from_logits(logits: torch.Tensor, masks: torch.BoolTensor) -> torch.Tensor:
    """Compute the entropy from logits.

    Args:
        logits: Logits tensor of shape (batch_size, sequence_length, vocab_size).
        masks: Boolean mask tensor of shape (batch_size, sequence_length) indicating valid positions.

    Returns:
        Entropy tensor of shape (batch_size, sequence_length).

    """

    logits, masks = logits[:, :-1], masks[:, 1:].to(logits.device)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)

    avg_entropy = (entropy * masks).sum() / (masks.sum() + 1e-3)

    return avg_entropy
