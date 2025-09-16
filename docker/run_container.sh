#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
# Runs an interactive bash within the container
# If you wish to keep the container, please do not use `--rm`

# NVIDIA
docker run --rm \
    --gpus all \
    --name phyagi-sdk \
    --shm-size=10g \
    --ipc=host \
    --volume ${HOME}/phyagi-sdk:/workspace/phyagi-sdk \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e NCCL_P2P_LEVEL=NVL \
    -it aifrontiers.azurecr.io/nvidia25.03-pytorch2.7.1-te2.4-deepspeed0.17.1-flashattn2.8.0.post2-vllm0.9.1:20250702

# AMD
# docker run --rm \
#     --device /dev/dri \
#     --device /dev/kfd \
#     --name phyagi-sdk \
#     --shm-size=10g \
#     --network=host \
#     --ipc=host \
#     --volume ${HOME}/phyagi-sdk:/workspace/phyagi-sdk \
#     --group-add video \
#     --cap-add SYS_PTRACE \
#     --security-opt seccomp=unconfined \
#     --privileged \
#     -it aifrontiers.azurecr.io/rocm6.3.4-pytorch2.7.1-deepspeed0.17.1-flashattn2.8.0.post2-vllm0.9.0.dev:20250702
