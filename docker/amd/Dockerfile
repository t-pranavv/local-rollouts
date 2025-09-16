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
ARG DEEPSPEED_VERSION
ARG FLASH_ATTENTION_VERSION
ARG VLLM_VERSION
ARG INSTALL_VLLM

# Installs desired packages for the workload
RUN apt-get update \
    && apt-get install flex redis redis-server --no-install-recommends --no-install-suggests -yq \
    && apt-get clean \
    && pip install --upgrade pip \
    && pip uninstall -y xgboost

# Update PyTorch (and related packages)
# `torchao==0.12.0` supports the 8-bit optimizer with FSDP
RUN ROCM_VERSION=$(basename $(readlink -f /opt/rocm) | sed -E 's/rocm-?//' | cut -d. -f1,2) \
    && pip uninstall -y torch torchvision torchaudio \
    && pip install torch==$TORCH_VERSION torchvision torchaudio --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}" \
    && pip install --pre torchao --index-url "https://download.pytorch.org/whl/nightly/rocm${ROCM_VERSION}"

# DeepSpeed and OpenMPI (Python)
RUN DS_BUILD_CPU_ADAGRAD=1 DS_BUILD_CPU_ADAM=1 DS_BUILD_CPU_LION=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LAMB=1 DS_BUILD_UTILS=1 \
    pip install deepspeed==$DEEPSPEED_VERSION \
    && CC=mpicc MPICC=mpicc pip install mpi4py --no-binary mpi4py

# Flash-Attention
# Temporarily out of `main` to avoid https://github.com/Dao-AILab/flash-attention/pull/1787
RUN MAX_JOBS=96 GPU_ARCHS="gfx942" pip install git+https://github.com/Dao-AILab/flash-attention --no-build-isolation

# vLLM
# Patch to use a forked version of vLLM that supports sleeping on AMD GPUs
RUN if [ "$INSTALL_VLLM" = "true" ]; then \
    git clone -b sleep_amd https://github.com/HollowMan6/vllm.git \
    && cd vllm \
    && ln -sf /opt/rocm/lib/libamdhip64.so /usr/lib/libamdhip64.so \
    && pip install /opt/rocm/share/amd_smi \
    && pip install --upgrade numba scipy huggingface-hub[cli,hf_transfer] setuptools_scm \
    && pip install "numpy<2" \
    && pip install -r requirements/rocm.txt \
    && pip install setuptools==78.1.0 \
    && MAX_JOBS=96 PYTORCH_ROCM_ARCH="gfx942" SETUPTOOLS_SCM_PRETEND_VERSION=$VLLM_VERSION python setup.py develop \
    && cd ..; \
fi

# Additional packages that prevents pip uninstalling when running on the cluster
RUN pip install --upgrade accelerate azure-identity azure-keyvault-secrets azure-storage-blob azureml-mlflow \
    datasets einops evaluate huggingface-hub lightning math-verify[antlr4_9_3] mlflow natsort omegaconf \
    pandas peft pre-commit pylatexenc pytest ray==2.44.1 redis safetensors tensorboard tokenizers transformers trl \
    typing-extensions wandb

# Ensures Redis configuration is overriden
RUN sed 's/bind 127.0.0.1 ::1/\# bind 127.0.0.1 ::1/g; s/protected-mode yes/protected-mode no/g' /etc/redis/redis.conf > ~/redis.conf \
    && cp ~/redis.conf /etc/redis/redis.conf

# Retrieves and install Singularity script
# Removing `/var/lib/apt/lists` or using `apt-get purge --auto-remove` breaks the PyTorch installation
COPY --from=installer /installer /opt/microsoft/_singularity/installations/
RUN sed -i '/rm -rf \/var\/lib\/apt\/lists\/\*/d' /opt/microsoft/_singularity/installations/common/env-setup.sh && \
    sed -i '/apt-get purge --auto-remove -y/d' /opt/microsoft/_singularity/installations/common/env-setup.sh
RUN sed -i '/rm -rf \/var\/lib\/apt\/lists\/\*/d' /opt/microsoft/_singularity/installations/common/run-install.sh && \
    sed -i '/apt-get purge --auto-remove /d' /opt/microsoft/_singularity/installations/common/run-install.sh
RUN /opt/microsoft/_singularity/installations/singularity/installer.sh

# Retrieves and validates Singularity script
COPY --from=validator /validations /opt/microsoft/_singularity/validations/
RUN /opt/microsoft/_singularity/validations/validator.sh

ENV HOME="/home/aiscuser"
ENV PATH="${HOME}/.local/bin:${PATH}"

RUN mkdir -p "${HOME}/.cache/torch/kernels"
RUN mkdir -p "${HOME}/.triton/autotune"
