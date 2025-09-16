# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from phyagi.datasets.concat_dataset import WeightedConcatIterableDataset
from phyagi.datasets.train.stream_lm.stream_lm_dataset import StreamLMDataset

CASES_SINGLE = [(8, 128, 3, [0.20, 0.30, 0.50])]
CASES_DIST = [(4, 8, 128, 3, [0.20, 0.30, 0.50])]


@pytest.fixture(scope="session", autouse=True)
def _init_distributed_process():
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:5000",
            rank=0,
            world_size=1,
        )
    yield
    dist.destroy_process_group()


def _get_rank_and_world():
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return dist.get_rank(), dist.get_world_size()


def _get_stream_lm_dataset(rank, world_size, *, num_datasets, n_seqs, seq_len):
    streams = []

    for i in range(num_datasets):
        raw = np.arange(n_seqs * seq_len, dtype=np.int32) + 1000 * i
        ds = StreamLMDataset(f"ds{i}", raw, seq_len=seq_len)
        ds.set_rank_and_world_size(rank, world_size)
        streams.append(ds)

    return streams


def _get_concat_dataset(rank, world_size, *, num_datasets, n_seqs, seq_len, weights, tracker=False):
    dataset = WeightedConcatIterableDataset(
        _get_stream_lm_dataset(
            rank,
            world_size,
            num_datasets=num_datasets,
            n_seqs=n_seqs,
            seq_len=seq_len,
        ),
        weights,
    )

    dataset.set_rank_and_world_size(rank, world_size)

    return dataset


def _sample_proportions(seq_len, n_seqs, num_datasets, weights):
    rank, world_size = _get_rank_and_world()

    dataset = _get_concat_dataset(
        rank,
        world_size,
        num_datasets=num_datasets,
        n_seqs=n_seqs,
        seq_len=seq_len,
        weights=weights,
        tracker=True,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)

    draws_per_rank = 20_000
    counts = np.zeros(num_datasets, dtype=np.int64)

    it = iter(loader)
    for _ in range(draws_per_rank):
        sample = next(it)
        counts[sample["dataset_idx"]] += 1

    tensor_counts = torch.as_tensor(counts)
    dist.all_reduce(tensor_counts, op=dist.ReduceOp.SUM)
    counts = tensor_counts.numpy()

    empirical = counts / counts.sum()
    expected = np.asarray(weights) / np.sum(weights)
    assert np.all(np.abs(empirical - expected) < 0.05)


def _restart_identical_batch(seq_len, n_seqs, num_datasets, weights):
    rank, world_size = _get_rank_and_world()

    steps_before_ckpt = 500
    steps_to_compare = 1000

    dataset_a = _get_concat_dataset(
        rank,
        world_size,
        num_datasets=num_datasets,
        n_seqs=n_seqs,
        seq_len=seq_len,
        weights=weights,
    )

    loader_a = torch.utils.data.DataLoader(dataset_a, batch_size=None, num_workers=0)
    iter_a = iter(loader_a)
    for _ in range(steps_before_ckpt):
        next(iter_a)

    saved_state = dataset_a.state_dict()
    ref_samples = [next(iter_a)["input_ids"].clone() for _ in range(steps_to_compare)]

    dataset_b = _get_concat_dataset(
        rank,
        world_size,
        num_datasets=num_datasets,
        n_seqs=n_seqs,
        seq_len=seq_len,
        weights=weights,
    )
    dataset_b.load_state_dict(saved_state)

    loader_b = torch.utils.data.DataLoader(dataset_b, batch_size=None, num_workers=0)
    iter_b = iter(loader_b)

    test_samples = [next(iter_b)["input_ids"].clone() for _ in range(steps_to_compare)]
    for (ref_sample, test_sample) in zip(ref_samples, test_samples):
        assert torch.equal(ref_sample, test_sample)


def _worker(rank, tmpfile, world_size, seq_len, n_seqs, num_datasets, weights, queue):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpfile}",
        rank=rank,
        world_size=world_size,
    )

    _sample_proportions(seq_len, n_seqs, num_datasets, weights)
    _restart_identical_batch(seq_len, n_seqs, num_datasets, weights)

    queue.put(True)

    dist.destroy_process_group()


@pytest.mark.parametrize("seq_len, n_seqs, num_datasets, weights", CASES_SINGLE)
def test_stream_lm_dataset_sample_proportions(seq_len, n_seqs, num_datasets, weights):
    _sample_proportions(seq_len, n_seqs, num_datasets, weights)


@pytest.mark.parametrize("seq_len, n_seqs, num_datasets, weights", CASES_SINGLE)
def test_stream_lm_dataset_restart_identical_batch(seq_len, n_seqs, num_datasets, weights):
    _restart_identical_batch(seq_len, n_seqs, num_datasets, weights)


@pytest.mark.skipif(os.name == "nt", reason="`init_method` is not supported on Windows.")
@pytest.mark.parametrize("world_size, seq_len, n_seqs, num_datasets, weights", CASES_DIST)
def test_stream_lm_dataset_multi_rank(world_size, seq_len, n_seqs, num_datasets, weights):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        rendezvous = Path(f.name)

    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()

    procs = [
        ctx.Process(
            target=_worker,
            args=(
                rank,
                rendezvous,
                world_size,
                seq_len,
                n_seqs,
                num_datasets,
                weights,
                queue,
            ),
        )
        for rank in range(world_size)
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()

    assert all(queue.get() for _ in range(world_size))
