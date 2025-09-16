# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from phyagi.datasets.rl.packing import distribute_and_pack_sequences


def _extract_sequences_from_packed(packed):
    tokens_flat = packed.tokens.flatten()
    masks_flat = packed.masks.flatten()
    advs_flat = packed.advantages.flatten()
    boundaries = packed.cu_seqlens.tolist()

    sequences = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        seq_tokens = tokens_flat[start:end].tolist()
        seq_masks = masks_flat[start:end].tolist()
        seq_advs = advs_flat[start:end].tolist()
        sequences.append((seq_tokens, seq_masks, seq_advs))
    return sequences


def _combine_gpu_batches(gpu_batches):
    all_sequences = []
    for gpu, batch_list in gpu_batches.items():
        for packed in batch_list:
            all_sequences.extend(_extract_sequences_from_packed(packed))
    return all_sequences


sequences = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 5, 6, 7, 8],
    [1, 2, 3, 4, 5, 6, 7],
    [1, 2, 3, 4, 5, 6],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4],
    [1, 2, 3],
    [1, 2],
    [1],
    [9],
    [10, 11],
    [10, 11, 12],
    [10, 11, 12, 13],
    [10, 11, 12, 13, 14],
    [10, 11, 12, 13, 14, 15],
    [10, 11, 12, 13, 14, 15, 16],
    [10, 11, 12, 13, 14, 15, 16, 17],
    [10, 11, 12, 13, 14, 15, 16, 17, 18],
]

assistant_masks = [
    [True, False, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, True],
    [True, False, False, False, False, True],
    [True, False, False, False, True],
    [True, False, False, True],
    [True, False, True],
    [True, True],
    [True],
    [True],
    [True, True],
    [True, False, True],
    [True, False, False, True],
    [True, False, False, False, True],
    [True, False, False, False, False, True],
    [True, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, True],
    [True, False, False, False, False, False, False, False, True],
]

advantages = [len(sequences) - i for i in range(len(sequences))]


@pytest.mark.parametrize("pad_to_largest_micro_batch", [False, True])
@pytest.mark.parametrize("micro_batch_size", [None, 2, 3])
@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_distribute_and_pack_sequences(micro_batch_size, world_size, pad_to_largest_micro_batch):
    max_length = 10
    gpu_batches = distribute_and_pack_sequences(
        sequences,
        assistant_masks,
        advantages,
        max_length,
        dp_size=world_size,
        tp_size=1,
        pad_token_id=-1,
        normalize_adv_num_tokens=False,
        micro_batch_size=micro_batch_size,
        pad_to_largest_micro_batch=pad_to_largest_micro_batch,
    )

    if pad_to_largest_micro_batch:
        assert len(set(len(batch) for batch in gpu_batches.values())) == 1

    extracted_sequences = _combine_gpu_batches(gpu_batches)
    num_sequences_found = 0

    for extracted_seq, extracted_mask, extracted_adv in extracted_sequences:
        if all(e == -1 for e in extracted_seq):
            assert all(e is False for e in extracted_mask)
            assert all(e == 0.0 for e in extracted_adv)
            continue

        assert extracted_seq in sequences
        match_index = sequences.index(extracted_seq)

        assert extracted_mask == assistant_masks[match_index]
        assert all(
            adv == advantages[match_index] if mask else adv == 0.0 for adv, mask in zip(extracted_adv, extracted_mask)
        )

        num_sequences_found += 1

    assert num_sequences_found == len(sequences)


def test_packing_even_distribution():
    packed = distribute_and_pack_sequences(
        sequences=sequences,
        assistant_masks=assistant_masks,
        advantages=advantages,
        max_length=9,
        pad_token_id=0,
        micro_batch_size=2,
        pad_to_largest_micro_batch=True,
        dp_size=2,
    )

    assert len(packed) == 2
    total_num_tokens = [sum(packed_b.tokens.numel() for packed_b in gpu_batch) for gpu_batch in packed.values()]
    assert total_num_tokens == [45, 45], "Tokens are not evenly distributed across data parallel ranks."
