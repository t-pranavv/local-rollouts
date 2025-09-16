# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# mpirun -np <num-processes> python mpi_vllm_generation.py

import argparse
import os
import subprocess
import sys
import time
from logging import StreamHandler, getLogger

from mpi4py import MPI
from openai import OpenAI

# Logger
logger = getLogger(__name__)
logger.setLevel("DEBUG")
logger.addHandler(StreamHandler(sys.stdout))

# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Constants
BASE_PORT = 8000
API_KEY = "key"
SERVER_HEARTBEAT = 60
SERVER_TIMEOUT = 720
SERVER_WAIT = 360


def _start_vllm_server(
    rank: int, pretrained_model_name_or_path: str, max_model_len: int, served_model_name: str
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    launch_cmd = [
        "vllm",
        "serve",
        pretrained_model_name_or_path,
        "--port",
        str(BASE_PORT + rank),
        "--api-key",
        API_KEY,
        "--max-model-len",
        str(max_model_len),
        "--served-model-name",
        served_model_name,
    ]

    logger.debug(f"Rank [{rank}]: {' '.join(launch_cmd)}")

    start_time = time.time()
    subprocess.Popen(launch_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    while time.time() - start_time < SERVER_TIMEOUT:
        time.sleep(SERVER_HEARTBEAT)

        response = subprocess.run(
            f"curl -s http://localhost:{BASE_PORT + rank}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if response.returncode == 0:
            logger.debug(f"Rank [{rank}]: Server is up and running...")

    logger.error(f"Rank [{rank}]: Shutting down server...")


def _run_inference(port: int, served_model_name: str) -> None:
    time.sleep(SERVER_WAIT)

    logger.info(f"Rank [{rank}]: Running inference on port {port}...")

    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key=API_KEY,
    )

    completion = client.chat.completions.create(
        model=served_model_name, messages=[{"role": "user", "content": "Hello!"}]
    )
    logger.info(f"Rank [{rank}]: {completion.choices[0].message}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launches vLLM server on some ranks and runs inference on others.")

    parser.add_argument("pretrained_model_name_or_path", type=str, help="Pre-trained model name or path.")

    parser.add_argument("served_model_name", type=str, help="Name of the served model.")

    parser.add_argument("-n", "--num_servers", type=int, default=1, help="Number of vLLM servers to spawn.")

    parser.add_argument("-m", "--max_model_len", type=int, default=8192, help="Maximum model length.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_args()

    server_ranks = list(range(args.num_servers))
    inference_ranks = list(range(args.num_servers, size))
    ports = [BASE_PORT + rank for rank in server_ranks]

    if rank in server_ranks:
        _start_vllm_server(rank, args.pretrained_model_name_or_path, args.max_model_len, args.served_model_name)

    if rank in inference_ranks:
        _run_inference(ports[rank % len(ports)], args.served_model_name)
