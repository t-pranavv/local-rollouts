#!/bin/bash

# Default values
ACR_NAME="aifrontiers"
IMAGE_NAME=""
TAG="latest"
DOCKERFILE_PATH=""
BUILD_CONTEXT="."
BUILD_ARGS=()

# Usage
usage () {
    # bash build_and_publish_docker.sh --dockerfile-path Dockerfile --image-name aif/reasoning-lm-tooluse --tag 220825 --build-context ./docker_context/
    echo "Usage: $0 --dockerfile-path <path> --image-name <image_name> [--acr-name <acr_name>] [--tag <tag>] [--build-context <path>] [--build-arg key=value]..."
    exit 1
}

azure_login () {
    echo "Logging in to Azure..."
    az account get-access-token --output json --resource https://management.azure.com
    if [ $? -eq 0 ]; then
        echo "Azure login successful"
    else
        az login --scope https://management.azure.com/.default --use-device-code
        if [ $? -eq 0 ]; then
            echo "Azure login successful"
        else
            echo "Azure login failed"
            sleep 3
            exit 1
        fi
    fi
}

acr_login () {
    echo "Logging in to Azure Container Registry '$ACR_NAME'..."
    az acr login --name "$ACR_NAME" --output none
    if [[ $? -ne 0 ]]; then
        echo "ACR login failed!"
        sleep 3
        exit 1
    fi
    echo "Successfully logged in to ACR: $ACR_NAME."
}

build_docker_image () {
    echo "Building Docker image: $IMAGE_NAME:$TAG..."

    registry="singularitybase"

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

    # Construct the build command with dynamic build arguments
    DOCKER_CMD="DOCKER_BUILDKIT=1 docker buildx build --platform linux/x86_64"

    for ARG in "${BUILD_ARGS[@]}"; do
        DOCKER_CMD+=" --build-arg $ARG"
    done
    DOCKER_CMD+=" --build-arg INSTALLER_IMAGE=$installer_image"
    DOCKER_CMD+=" --build-arg VALIDATOR_IMAGE=$validator_image"
    DOCKER_CMD+=" -t $ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG -f $DOCKERFILE_PATH $BUILD_CONTEXT --progress=plain"

    # Execute the constructed command
    eval $DOCKER_CMD 2>&1 | tee build.file

    if [[ $? -ne 0 ]]; then
        echo "Docker build failed!"
        sleep 3
        exit 1
    fi
    echo "Docker image built: $ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG."
}

publish_docker_image () {
    echo "Pushing Docker image to ACR: $ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG..."
    docker push "$ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG"
    if [[ $? -ne 0 ]]; then
        echo "Docker push failed!"
        sleep 3
        exit 1
    fi
    echo "Docker image published: $ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG."
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --acr-name)
            ACR_NAME="$2"
            shift 2
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --dockerfile-path)
            DOCKERFILE_PATH="$2"
            shift 2
            ;;
        --build-context)
            BUILD_CONTEXT="$2"
            shift 2
            ;;
        --build-arg)
            BUILD_ARGS+=("$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate the required parameters
if [[ -z "$IMAGE_NAME" || -z "$DOCKERFILE_PATH" ]]; then
    echo "Error: --image-name and --dockerfile-path are required."
    usage
fi

azure_login
acr_login
build_docker_image
publish_docker_image

echo "Docker image successfully built and published to Azure Container Registry!"