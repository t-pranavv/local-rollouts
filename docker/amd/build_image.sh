#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Constants
registry="singularitybase"
base_image="rocm/pytorch:rocm6.3.4_ubuntu22.04_py3.10_pytorch_release_2.4.0"

torch_version="2.7.1"
deepspeed_version="0.17.4"
flash_attention_version="2.8.2"
vllm_version="0.10.0.dev"

install_vllm=false
vllm_tag=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --install-vllm)
            install_vllm=true
            vllm_tag="-vllm$vllm_version"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Creates the validation image
validator_image_repo="validations/base/singularity-tests"
validator_image_tag=`az acr manifest list-metadata \
    --registry $registry \
    --name $validator_image_repo \
    --orderby time_desc \
    --query '[].{Tag:tags[0]}' \
    --output tsv \
    --top 1`
validator_image=$registry.azurecr.io/$validator_image_repo:${validator_image_tag%%[[:cntrl:]]}

# Creates the installer image
installer_image_repo="installer/base/singularity-installer"
installer_image_tag=`az acr manifest list-metadata \
    --registry $registry \
    --name $installer_image_repo \
    --orderby time_desc \
    --query '[].{Tag:tags[0]}' \
    --output tsv \
    --top 1`
installer_image=$registry.azurecr.io/$installer_image_repo:${installer_image_tag%%[[:cntrl:]]}

# Infers current date, and versions from the base image
current_date=$(date +"%Y%m%d")
rocm_version=$(echo "$base_image" | sed -E 's|.*rocm([0-9]+\.[0-9]+\.[0-9]+).*|\1|')

# Builds the custom image
output_image_tag="aifrontiers.azurecr.io/rocm${rocm_version}-pytorch${torch_version}-deepspeed${deepspeed_version}-flashattn${flash_attention_version}-vllm${vllm_version}:${current_date}"
docker build . \
    --file Dockerfile \
    --tag $output_image_tag \
    --build-arg BASE_IMAGE=$base_image \
    --build-arg INSTALLER_IMAGE=$installer_image \
    --build-arg VALIDATOR_IMAGE=$validator_image \
    --build-arg TORCH_VERSION=$torch_version \
    --build-arg DEEPSPEED_VERSION=$deepspeed_version \
    --build-arg FLASH_ATTENTION_VERSION=$flash_attention_version \
    --build-arg VLLM_VERSION=$vllm_version \
    --build-arg INSTALL_VLLM=$install_vllm