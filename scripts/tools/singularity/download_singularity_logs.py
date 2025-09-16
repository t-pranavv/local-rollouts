# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import multiprocessing
import os
from pathlib import Path


# Download logs from a node using `scp`
def _download_log(node_id: str, node_num: str, dir_name: Path) -> None:
    command = 'scp -i ~/.ssh/id_rsa \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ProxyCommand="ssh -i ~/.ssh/id_rsa -W %h:%p \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        aiscuser@aisc-prod-eastus2-cpml-lnx-7.federation.singularity.azure.com" \
        aiscuser@node-{}.{}:/scratch/azureml/cr/j/*/exe/wd/user_logs/std_log_process* \
        {}/.'.format(
        node_num, node_id, str(dir_name)
    )

    os.system(command)


if __name__ == "__main__":
    # Base node ID (retrieve from Debug and Monitor tab on AML Studio)
    node_id = "185d6846-79fb-4566-97a8-806d62914007"
    num_nodes = 64

    # Create directory to store logs
    dir_path = Path("/tmp/data") / "logs-{}".format(node_id)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Loop over nodes from job and download user logs
    num_threads = 8
    with multiprocessing.Pool(num_threads) as pool:
        pool.starmap(_download_log, [(node_id, i, dir_path) for i in range(num_nodes)])
