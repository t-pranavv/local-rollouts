# Copyright (c) https://gist.github.com/jeffra/b5e80466b4c86be00ea3b6f130fb7a36.
# Licensed under the MIT license.

import argparse
import os
import time
from typing import Callable, Tuple

import deepspeed
import torch
import torch.distributed as dist

# 6GB
N = 500000
M = 500 * 6
TRIALS = 5


def _timed_allreduce(mat: torch.Tensor) -> Tuple[float, float]:
    torch.cuda.synchronize()
    pre = time.perf_counter()

    dist.all_reduce(mat)

    torch.cuda.synchronize()
    duration = time.perf_counter() - pre

    size = M * N * 4
    n = dist.get_world_size()

    tput = ((M * N * 4 * 2) / duration) * 8
    busbw = (size / duration) * (2 * (n - 1) / n) * 8

    return tput, busbw


def _run(local_rank: int) -> None:
    global_rank = dist.get_rank()
    if global_rank == 0:
        print(global_rank, "data size:", M * N * 4 / 1e9, "GB")

    mat = torch.rand(N, M, dtype=torch.float32).cuda(local_rank)
    tputs, busbws = [], []

    for trial in range(TRIALS):
        tput, busbw = _timed_allreduce(mat)
        if trial > 2:
            tputs.append(tput)
            busbws.append(busbw)

    local_avg = sum(tputs) / len(tputs)
    local_avg_bb = sum(busbws) / len(busbws)

    t = torch.tensor([local_avg / 1e9, local_avg_bb / 1e9], device="cuda")
    dist.all_reduce(t)

    tput_avg = t[0] / dist.get_world_size()
    busbw_avg = t[1] / dist.get_world_size()

    print("tput_avg (Gbps):", tput_avg.item(), "busbw_avg (Gbps):", busbw_avg.item())

    dist.barrier()


def _init_processes(local_rank: int, fn: Callable, backend: str = "nccl") -> None:
    deepspeed.init_distributed(dist_backend=backend)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    fn(local_rank)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Performs an all-reduce test.")

    parser.add_argument("local_rank", type=int, help="Local rank of the process.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_args()
    rank = args.local_rank

    _init_processes(local_rank=rank, fn=_run)
