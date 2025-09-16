#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Environment variables
PYTHON_MIN_VERSION=${PYTHON_MIN_VERSION:-"3.10"}
CUDA_VERSION=${CUDA_VERSION:-"12.8"}
PYTORCH_VERSION=${PYTORCH_VERSION:-"2.7.1"}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-"0.22.1"}
VLLM_VERSION=${VLLM_VERSION:-"0.9.1"}
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
LOG_FILE="$SCRIPT_DIR/install_log.txt"

print_message() {
    printf "\033[1;32m%s\033[0m\n" "$1"
}

check_python() {
    if ! command -v python &> /dev/null; then
        print_message "Python is not installed. Please install Python $PYTHON_MIN_VERSION+."
        exit 1
    fi

    local python_version
    python_version=$(python -V 2>&1 | awk '{print $2}')

    local python_major_version
    local python_minor_version
    python_major_version=$(echo "$python_version" | cut -d. -f1)
    python_minor_version=$(echo "$python_version" | cut -d. -f2)

    local required_major_version
    local required_minor_version
    required_major_version=$(echo "$PYTHON_MIN_VERSION" | cut -d. -f1)
    required_minor_version=$(echo "$PYTHON_MIN_VERSION" | cut -d. -f2)

    # Compare versions numerically
    if ((python_major_version < required_major_version)) ||
       ((python_major_version == required_major_version && python_minor_version < required_minor_version)); then
        print_message "Python version must be $PYTHON_MIN_VERSION or higher. Found: $python_version"
        exit 1
    fi
}

install_cuda() {
    print_message "Installing CUDA version $CUDA_VERSION via Anaconda..."
    conda install cuda -c "nvidia/label/cuda-${CUDA_VERSION}" -y
}

install_pytorch() {
    print_message "Installing PyTorch version $PYTORCH_VERSION..."
    if command -v nvidia-smi &> /dev/null; then
        # System has GPU support
        pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION --index-url $PYTORCH_INDEX_URL
    else
        # Install CPU-only PyTorch
        pip install torch==$PYTORCH_VERSION torchvision==$TORCHVISION_VERSION
    fi
}

setup_environment_variables() {
    if [ -z "$DATA_ROOT" ]; then
        print_message "Setting DATA_ROOT to /mnt/phyagi..."
        echo -e '\nexport DATA_ROOT="/mnt/phyagi"' >> ~/.bashrc
        source ~/.bashrc
    else
        print_message "DATA_ROOT is already set to $DATA_ROOT"
    fi

    if [ -z "$WANDB_HOST" ]; then
        print_message "Setting WANDB_HOST to https://microsoft-research.wandb.io..."
        echo -e 'export WANDB_HOST="https://microsoft-research.wandb.io"' >> ~/.bashrc
        source ~/.bashrc
    else
        print_message "WANDB_HOST is already set to $WANDB_HOST"
    fi
}

install_dependencies() {
    print_message "Upgrading pip and installing basic packages..."
    pip install --upgrade pip
    pip install numpy packaging ninja
    conda install mpi4py -y
}

install_phyagi_sdk() {
    print_message "Installing PhyAGI..."
    pip install -e .
}

install_vllm() {
    print_message "Installing vLLM..."
    pip install vllm
}

install_additional_dependencies() {
    case $1 in
        "eval")
            print_message "Installing evaluation packages..."
            pip install -e .[eval]
            ;;
        "flash-attn")
            print_message "Installing Flash-Attention..."
            bash "${SCRIPT_DIR}/scripts/tools/install/flash-attn.sh"
            pip install -e .[flash-attn]
            ;;
        "rl")
            print_message "Installing reinforcement learning packages..."
            pip install -e .[rl]
            ;;
        *)
            print_message "Unknown option: $1"
            ;;
    esac
}

main_installation() {
    check_python
    setup_environment_variables
    install_dependencies

    # vLLM installation breaks previous installations, so we install it first
    # and then override installed versions with the ones from phyagi-sdk
    while true; do
        read -p "Do you want to install vLLM? (y/n): " yn
        case $yn in
            [Yy]* ) install_vllm; break;;
            [Nn]* ) break;;
            * ) echo "Please answer yes or no.";;
        esac
    done

    while true; do
        read -p "Do you want to install CUDA? (y/n): " yn
        case $yn in
            [Yy]* ) install_cuda; break;;
            [Nn]* ) break;;
            * ) echo "Please answer yes or no.";;
        esac
    done

    while true; do
        read -p "Do you want to install PyTorch? (y/n): " yn
        case $yn in
            [Yy]* ) install_pytorch; break;;
            [Nn]* ) break;;
            * ) echo "Please answer yes or no.";;
        esac
    done

    install_phyagi_sdk

    for component in "eval" "flash-attn" "rl"; do
        read -p "Do you want to install $component? (y/n): " yn
        if [[ $yn =~ ^[Yy]$ ]]; then
            install_additional_dependencies "$component"
        fi
    done
}

main_installation | tee "$LOG_FILE"
