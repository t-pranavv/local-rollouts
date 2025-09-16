#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit on error
set -e

# Set variables
FOLDER_NAME="flash-attention"
GITHUB_REPO="https://github.com/Dao-AILab/flash-attention.git"
REPO_VERSION="v2.8.0.post2"
SUBREPOS=("csrc/fused_dense_lib")
CURRENT_DIR=$(pwd)

# Clone the main repository and checkout specific version
rm -rf "$FOLDER_NAME"
git clone --depth=1 --branch "$REPO_VERSION" "$GITHUB_REPO" "$FOLDER_NAME" --single-branch
cd "$FOLDER_NAME"

# Install the main project (`setup.py` is preferable for better compatibility)
# `pip install .` will force package override if the package is already installed
echo "Installing '$FOLDER_NAME'..."
python setup.py install
pip install .

# Install subrepositories using `pip`
for subrepo in "${SUBREPOS[@]}"; do
    echo "Installing '$subrepo'..."
    cd "$subrepo"
    pip install .
    cd "$CURRENT_DIR/$FOLDER_NAME"
done

# Remove the `flash-attention` directory
cd "$CURRENT_DIR"
rm -rf "$FOLDER_NAME"

echo "Installation completed."