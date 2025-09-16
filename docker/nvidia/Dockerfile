# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Image-related arguments
ARG BASE_IMAGE
ARG INSTALLER_IMAGE
ARG VALIDATOR_IMAGE

FROM --platform=linux/amd64 $BASE_IMAGE AS base
FROM --platform=linux/amd64 $INSTALLER_IMAGE AS installer
FROM --platform=linux/amd64 $VALIDATOR_IMAGE AS validator

# Base image
FROM base

# Non-image-related arguments
# Must be defined after loading the base image to avoid losing their values
ARG TORCH_VERSION
ARG TRANSFORMER_ENGINE_VERSION
ARG DEEPSPEED_VERSION
ARG FLASH_ATTENTION_VERSION
ARG VLLM_VERSION
ARG INSTALL_VLLM

# NVIDIA base images started creating a default constraint file for `pip`
# and that blocks re-installation of some packages
ENV PIP_CONSTRAINT=""

# Installs desired packages for the workload
RUN apt-get update \
    && apt-get install redis redis-server --no-install-recommends --no-install-suggests -yq \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge --auto-remove \
    && apt-get clean \
    && pip install --upgrade pip \
    && pip uninstall -y xgboost

# Update PyTorch (and related packages)
# `torchao==0.12.0` supports the 8-bit optimizer with FSDP
RUN CUDA_VERSION=$(nvcc --version | grep 'release' | sed -E 's/.*release ([0-9]+)\.([0-9]+),.*/cu\1\2/') \
    && pip install torch==$TORCH_VERSION torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}" \
    && pip install --pre torchao --index-url "https://download.pytorch.org/whl/nightly/${CUDA_VERSION}"

# Update TransformerEngine
RUN pip uninstall -y transformer-engine \
    && git clone --branch release_v${TRANSFORMER_ENGINE_VERSION} --recursive https://github.com/NVIDIA/TransformerEngine.git \
    && cd TransformerEngine \
    && export NVTE_FRAMEWORK=pytorch \
    && pip install . \
    && cd .. && rm -rf TransformerEngine

# DeepSpeed and OpenMPI (Python)
RUN DS_BUILD_CPU_ADAGRAD=1 DS_BUILD_CPU_ADAM=1 DS_BUILD_CPU_LION=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LAMB=1 DS_BUILD_UTILS=1 \
    pip install deepspeed==$DEEPSPEED_VERSION \
    && CC=mpicc MPICC=mpicc pip install mpi4py --no-binary mpi4py

# Flash-Attention and CUDA extension for fused dense
RUN git clone https://github.com/Dao-AILab/flash-attention \
    && cd flash-attention && git checkout v$FLASH_ATTENTION_VERSION \
    && python setup.py install \
    && FLASH_ATTENTION_FORCE_BUILD=TRUE pip install . \
    && cd csrc/fused_dense_lib && pip install . && cd ../../..

# Flash-Attention 3 (for Hopper GPUs)
# Temporary out of `main` due to https://github.com/Dao-AILab/flash-attention/pull/1719
RUN cd flash-attention/hopper && git checkout main \
    && python setup.py install \
    && FLASH_ATTENTION_FORCE_BUILD=TRUE pip install . \
    && cd .. && rm -rf flash-attention

# vLLM
RUN if [ "$INSTALL_VLLM" = "true" ]; then \
    git clone https://github.com/vllm-project/vllm.git \
    && cd vllm && git checkout v$VLLM_VERSION \
    && python use_existing_torch.py \
    && pip install -r requirements/build.txt \
    && MAX_JOBS=96 pip install . --no-build-isolation \
    && cd .. && rm -rf vllm; \
fi

# Additional packages that prevents pip uninstalling when running on the cluster
RUN pip install --upgrade accelerate azure-identity azure-keyvault-secrets azure-storage-blob azureml-mlflow \
    datasets einops evaluate huggingface-hub lightning math-verify[antlr4_9_3] mlflow natsort omegaconf \
    pandas peft pre-commit pylatexenc pytest ray redis safetensors tensorboard tokenizers transformers trl \
    typing-extensions wandb

# Ensures Redis configuration is overriden
RUN sed 's/bind 127.0.0.1 ::1/\# bind 127.0.0.1 ::1/g; s/protected-mode yes/protected-mode no/g' /etc/redis/redis.conf > ~/redis.conf \
    && cp ~/redis.conf /etc/redis/redis.conf

# Retrieves and install Singularity script
COPY --from=installer /installer /opt/microsoft/_singularity/installations/
RUN /opt/microsoft/_singularity/installations/singularity/installer.sh

# Sets some validation environment variables for additional checks
ENV SINGULARITY_IMAGE_ACCELERATOR="NVIDIA"

# Retrieves and validates Singularity script
COPY --from=validator /validations /opt/microsoft/_singularity/validations/
RUN /opt/microsoft/_singularity/validations/validator.sh

ENV HOME="/home/aiscuser"
ENV PATH="${HOME}/.local/bin:${PATH}"

RUN mkdir -p "${HOME}/.cache/torch/kernels"
RUN mkdir -p "${HOME}/.triton/autotune"
