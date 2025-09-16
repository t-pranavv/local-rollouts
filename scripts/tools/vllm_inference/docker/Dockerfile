# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# DOCKER_BUILDKIT=1 docker buildx build --platform linux/x86_64 -t "aif/reasoning-lm:220825" -f Dockerfile ./docker_context/ --progress=plain

# Arguments supplied by running script
ARG BASE_IMAGE="nvcr.io/nvidia/pytorch:25.03-py3"
ARG INSTALLER_IMAGE
ARG VALIDATOR_IMAGE

# Creates variables from arguments
FROM --platform=linux/amd64 $BASE_IMAGE AS base
FROM --platform=linux/amd64 $INSTALLER_IMAGE AS installer
FROM --platform=linux/amd64 $VALIDATOR_IMAGE AS validator

# Defines the base image
FROM base

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive
ARG TARGETPLATFORM='linux/x86_64'
ARG torch_cuda_arch_list='9.0 10.0 12.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

# NVIDIA base images started creating a default constraint file for `pip`
# and that blocks re-installation of some packages
ENV PIP_CONSTRAINT=""

# Retrieves and install Singularity script
COPY --from=installer /installer /opt/microsoft/_singularity/installations/
RUN /opt/microsoft/_singularity/installations/singularity/installer.sh

# Sets some validation environment variables for additional checks
ENV SINGULARITY_IMAGE_ACCELERATOR="NVIDIA"

# Retrieves and validates Singularity script
COPY --from=validator /validations /opt/microsoft/_singularity/validations/
RUN /opt/microsoft/_singularity/validations/validator.sh

RUN echo "aiscuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Add Permissions
RUN chown -R :aiscuser /workspace /usr/local/lib/python3.12 /usr/local/bin/ /usr/include/python3.12 /usr/lib/python3/dist-packages
RUN chmod -R 2775 /workspace /usr/local/lib/python3.12 /usr/local/bin/ /usr/include/python3.12 /usr/lib/python3/dist-packages

# Installation/Basic Utilities
RUN apt-get update && apt-get install -yq --no-install-recommends \
        bc redis redis-server software-properties-common build-essential autotools-dev \
        pdsh g++ gcc curl wget vim tmux emacs less unzip htop iftop iotop ca-certificates \
        rsync iputils-ping net-tools sudo libfuse-dev fuse git git-lfs libnuma-dev \
        dos2unix psmisc graphviz llvm-dev ninja-build npm libaio-dev jq lshw dmidecode \
        util-linux automake autoconf libtool perftest net-tools openssh-client openssh-server \
        pciutils libaio-dev libcap2 default-jdk lsb-release gnupg squashfs-tools uidmap \
    && add-apt-repository -y ppa:apptainer/ppa && apt-get update \
    && apt-get purge -yq python3-blinker \
    && apt-get install -y --no-install-recommends apptainer-suid \
    && rm -rf /var/lib/apt/lists/* && apt-get purge -yq --auto-remove && apt-get clean -yq \
    && pip install --upgrade pip \
    && pip install 'cmake>=3.26' 'setuptools>=61' 'setuptools-scm>=8' packaging pybind11 ninja wheel \
    && apptainer --version

# Ensures Redis configuration is overriden
RUN sed 's/bind 127.0.0.1 ::1/\# bind 127.0.0.1 ::1/g; s/protected-mode yes/protected-mode no/g' /etc/redis/redis.conf > /home/aiscuser/redis.conf \
    && cp /home/aiscuser/redis.conf /etc/redis/redis.conf

# Install AZ-Cli
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash && \
    az --version

# Install Azcopy
RUN wget -O azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux && \
    tar -xf azcopy.tar.gz && \
    sudo cp azcopy_linux_amd64_*/azcopy /usr/bin/azcopy && \
    rm -rf azcopy.tar.gz azcopy_linux_amd64_* && \
    azcopy --version

# Switch to aiscuser for all package installation
USER aiscuser

# Setup paths
ENV HOME="/home/aiscuser"
ENV PATH="${HOME}/.local/bin:${PATH}"
ENV APPTAINER_HOME="${HOME}/apptainer"
RUN mkdir -p "${HOME}/apptainer"
RUN mkdir -p "${HOME}/.cache/torch/kernels"
RUN mkdir -p "${HOME}/.triton/autotune"

# Setup TikToken Cache
RUN mkdir -p /workspace/tiktoken_cache
RUN mkdir -p /workspace/tiktoken_encodings

ARG TIKTOKEN_URL="https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
RUN wget -O /workspace/tiktoken_cache/$(echo -n $TIKTOKEN_URL | sha1sum | head -c 40) $TIKTOKEN_URL
RUN wget -O /workspace/tiktoken_encodings/cl100k_base.tiktoken $TIKTOKEN_URL

# Setup GPT-OSS Harmany TikToken Cache
ARG TIKTOKEN_URL="https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
RUN wget -O /workspace/tiktoken_cache/$(echo -n $TIKTOKEN_URL | sha1sum | head -c 40) $TIKTOKEN_URL
RUN wget -O /workspace/tiktoken_encodings/o200k_base.tiktoken $TIKTOKEN_URL

ENV TIKTOKEN_CACHE_DIR=/workspace/tiktoken_cache
ENV TIKTOKEN_RS_CACHE_DIR=/workspace/tiktoken_cache
ENV TIKTOKEN_ENCODINGS_BASE=/workspace/tiktoken_encodings

RUN pip install nltk
# Setup NLTK Cache
RUN mkdir -p /workspace/nltk_data
RUN python -m nltk.downloader -d /workspace/nltk_data all
ENV NLTK_DATA=/workspace/nltk_data

# Add Permissions
RUN sudo chown -R :aiscuser /usr/local/share/man
RUN sudo chmod -R 2775 /usr/local/share/man

# Install torch && torchao
RUN CUDA_VERSION=$(nvcc --version | grep 'release' | sed -E 's/.*release ([0-9]+)\.([0-9]+),.*/cu\1\2/') \
    && pip install torch==2.7.1 torchvision torchao --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

# Update TransformerEngine
ENV NVTE_FRAMEWORK='pytorch'
RUN pip uninstall -y transformer-engine \
    && git clone --recursive https://github.com/NVIDIA/TransformerEngine.git \
    && cd TransformerEngine && git checkout tags/v2.5 && git submodule update --init --recursive \
    && pip install . && cd .. \
    && rm -rf TransformerEngine

# Install Apex (required for Megatron)
RUN MAX_JOBS=96 NINJA_FLAGS="-j96" pip install --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex

# DeepSpeed and OpenMPI (Python)
RUN DS_BUILD_CPU_ADAGRAD=1 DS_BUILD_CPU_ADAM=1 DS_BUILD_CPU_LION=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LAMB=1 DS_BUILD_UTILS=1 \
    pip install deepspeed=="0.17.4" \
    && CC=mpicc MPICC=mpicc pip install mpi4py --no-binary mpi4py

# Install FlashInfer
RUN git clone https://github.com/flashinfer-ai/flashinfer.git --recursive \
    && cd flashinfer && git checkout tags/v0.2.13 && git submodule update --init --recursive \
    && MAX_JOBS=96 NINJA_FLAGS="-j96" FLASHINFER_ENABLE_AOT=1 pip install . && cd .. \
    && rm -rf flashinfer

# Flash-Attention and CUDA extensions for fused dense, layer norm, etc
ENV FLASH_ATTENTION_VERSION="2.8.3"
RUN git clone https://github.com/Dao-AILab/flash-attention \
    && cd flash-attention && git checkout v$FLASH_ATTENTION_VERSION \
    && python setup.py install \
    && FLASH_ATTENTION_FORCE_BUILD=TRUE pip install . \
    && cd csrc/layer_norm && pip install . && cd ../../ \
    && cd csrc/fused_dense_lib && pip install . && cd ../../ && cd .. \
    && rm -rf flash-attention

# vLLM
ENV VLLM_VERSION="0.10.1.1"
RUN git clone https://github.com/vllm-project/vllm.git \
    && cd vllm && git checkout v$VLLM_VERSION \
    && python use_existing_torch.py \
    && pip install -r requirements/build.txt \
    && MAX_JOBS=96 NINJA_FLAGS="-j96" pip install . --no-build-isolation && cd .. \
    && rm -rf vllm

# Additional packages that prevents pip uninstalling when running on the cluster
RUN pip install accelerate 'azure-identity==1.17.0' azure-storage-blob azureml-mlflow datasets einops evaluate \
    huggingface-hub liger-kernel lightning mlflow natsort omegaconf pandas peft pre-commit pytest redis \
    safetensors tensorboard tiktoken tokenizers 'transformers==4.55.2' trl typing-extensions wandb \
    'azure-ai-ml>=1.14.0' 'azureml-core>=1.56.0' azure-keyvault-secrets boto3 bitsandbytes codetiming dill \
    Faker hydra-core modelscope numpy packaging peft pyarrow pybind11 pylatexenc py-spy runai-model-streamer \
    runai-model-streamer[s3] safetensors tensordict timm yapf latex2sympy2-extended hf_transfer h5py tqdm openai \
    azure-ai-evaluation virtualenv prettytable joblib math-verify[antlr4_9_3] ray[default] torchdata tenacity uvicorn \
    fastapi langchain-azure-dynamic-sessions orjson mcp[cli] gpt-oss
ENV HF_HUB_ENABLE_HF_TRANSFER 1

# Phigen - Commit: 557aa8b20ee4742f3aa8a342ca41d9cfb219985e
COPY --chown=aiscuser:aiscuser ./phigen-1.2.0-py3-none-any.whl /workspace/phigen-1.2.0-py3-none-any.whl

# Phyagi-SDK - Commit: 2c68aec6181ec16accc13604505d90cf6bdc4184
COPY --chown=aiscuser:aiscuser ./phyagi-3.2.2.dev0-py3-none-any.whl /workspace/phyagi-3.2.2.dev0-py3-none-any.whl

# Install wheels
RUN pip install /workspace/*.whl

USER root