# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import subprocess
import sys
import time
from argparse import Namespace, _SubParsersAction
from typing import Dict, Optional

from phyagi.utils.logging_utils import get_logger

logger = get_logger(__name__)


class StartRayClusterCommand:
    """CLI-based command for launching a Ray cluster for distributed jobs."""

    _WORKERS_START_WAIT_TIME = 60
    _HEAD_START_WAIT_TIME = 10

    def register(parser: _SubParsersAction) -> None:
        """Register the command with the given parser.

        Args:
            parser: Parser to register the command with.

        """

        command_parser = parser.add_parser("start-ray-cluster", help="Starts a Ray cluster for multi-node jobs.")

        command_parser.add_argument(
            "--node_rank", type=int, default=int(os.getenv("NODE_RANK", "0")), help="Rank of the node (0 for head)."
        )

        command_parser.add_argument(
            "--head_addr", type=str, default=os.getenv("MASTER_ADDR"), help="Address of the head node."
        )

        command_parser.add_argument(
            "--head_port", type=str, default=os.getenv("MASTER_PORT"), help="Port for the head node."
        )

        command_parser.set_defaults(func=StartRayClusterCommand)

    def __init__(self, args: Namespace, extra_args: Optional[Dict[str, any]] = None) -> None:
        """Initialize the command with the given arguments.

        Args:
            args: Arguments passed to the command.
            extra_args: Extra arguments passed to the command (not captured by the parser).

        """

        self._node_rank = args.node_rank
        self._head_addr = args.head_addr
        self._head_port = args.head_port

        if self._head_port is None:
            raise ValueError(
                "`head_port` must be provided. Set the environment variable `MASTER_PORT` or the `--head_port` argument."
            )
        if self._node_rank != 0 and self._head_addr is None:
            raise ValueError(
                "`head_addr` must be provided. Set the environment variable `MASTER_ADDR` or the `--head_addr` argument."
            )

    def _start_ray_head(self) -> None:
        cmd = ["ray", "start", "--head", f"--port={self._head_port}"]
        logger.info(f"Starting head: {' '.join(cmd)}")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f"Error when starting head: {stderr}")
            sys.exit(1)

        logger.info(stdout)
        logger.info("Head started successfully.")

    def _start_ray_worker(self) -> None:
        ray_address = f"{self._head_addr}:{self._head_port}"
        cmd = ["ray", "start", f"--address={ray_address}", "--block"]

        logger.info(f"Starting worker: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:
            # Block and keep the worker alive
            _, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"Error when starting worker: {stderr}")
                sys.exit(1)
        except KeyboardInterrupt:
            logger.warning("Terminating worker...")
            process.terminate()
            process.wait()

    def _check_ray_status(self) -> None:
        try:
            result = subprocess.run(["ray", "status"], capture_output=True, text=True, check=True)
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error when checking status: {e.stderr}")

    def run(self) -> None:
        """Run the command."""

        if self._node_rank == 0:
            self._start_ray_head()
            logger.info("Waiting for workers to start...")
            time.sleep(self._WORKERS_START_WAIT_TIME)
        else:
            logger.info("Waiting for head to start...")
            time.sleep(self._HEAD_START_WAIT_TIME)
            self._start_ray_worker()

        self._check_ray_status()
