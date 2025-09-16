#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Exit on error
set -e

# Set variables
FOLDER_NAME="llama.cpp"
GITHUB_REPO="https://github.com/ggerganov/llama.cpp.git"
CURRENT_DIR=$(pwd)
BUILD_DIR="build"
INSTALL_BIN_DIR="$HOME/.local/bin"
INSTALL_LIB_DIR="$HOME/.local/lib"

# Check for environment variables
if [[ ":$PATH:" != *":$INSTALL_BIN_DIR:"* ]]; then
    echo "Adding $INSTALL_BIN_DIR to PATH in ~/.bashrc..."
    echo -e "\nexport PATH=\"\$HOME/.local/bin:\$PATH\"" >> ~/.bashrc
    export PATH="$INSTALL_BIN_DIR:$PATH"
else
    echo "$INSTALL_BIN_DIR is already in PATH."
fi

if [[ ":$LD_LIBRARY_PATH:" != *":$INSTALL_LIB_DIR:"* ]]; then
    echo "Adding $INSTALL_LIB_DIR to LD_LIBRARY_PATH in ~/.bashrc..."
    echo -e "\nexport LD_LIBRARY_PATH=\"\$HOME/.local/lib:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
    export LD_LIBRARY_PATH="$INSTALL_LIB_DIR:$LD_LIBRARY_PATH"
else
    echo "$INSTALL_LIB_DIR is already in LD_LIBRARY_PATH."
fi

# Clone the main repository
mkdir -p "$INSTALL_BIN_DIR"
rm -rf "$FOLDER_NAME"
git clone "$GITHUB_REPO" "$FOLDER_NAME"
cd "$FOLDER_NAME"

# Check for `libcurl` development files
echo "Checking for 'libcurl'..."
if pkg-config --exists libcurl; then
    echo "'libcurl' found. Enabling support..."
    CURL_FLAG="-DLLAMA_CURL=ON"
else
    echo "'libcurl' not found. Attempting to install (Debian-based systems only)..."
    if command -v apt &> /dev/null; then
        sudo apt update
        sudo apt install -y libcurl4-openssl-dev || {
            echo "Failed to install 'libcurl'. Disabling support..."
            CURL_FLAG="-DLLAMA_CURL=OFF"
        }
    else
        echo "Not a Debian-based system or 'apt' not found. Disabling support..."
        CURL_FLAG="-DLLAMA_CURL=OFF"
    fi
fi

# Build with CMake
echo "Building 'llama.cpp'..."
cmake $CURL_FLAG -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --config Release
pip install -r requirements/requirements-convert_legacy_llama.txt

# Move `llama.cpp` binaries and shared libraries to proper directories
echo "Moving 'llama.cpp' binaries to $INSTALL_BIN_DIR..."

for bin in $BUILD_DIR/bin/*; do
    echo "Moving $bin to $INSTALL_BIN_DIR"
    [ -f "$bin" ] && install -m 755 "$bin" "$INSTALL_BIN_DIR/"
done

echo "Moving 'llama.cpp' shared libraries to $INSTALL_LIB_DIR..."
for lib in $BUILD_DIR/bin/*.so; do
    echo "Moving $lib to $INSTALL_LIB_DIR"
    [ -f "$lib" ] && install -m 755 "$lib" "$INSTALL_LIB_DIR/"
done

echo "Moving 'convert_hf_to_gguf.py' to $INSTALL_BIN_DIR..."
install -m 755 convert_hf_to_gguf.py "$INSTALL_BIN_DIR/convert_hf_to_gguf.py"
if ! head -n 1 "$INSTALL_BIN_DIR/convert_hf_to_gguf.py" | grep -q python; then
    sed -i '1i#!/usr/bin/env python3' "$INSTALL_BIN_DIR/convert_hf_to_gguf.py"
fi
chmod +x "$INSTALL_BIN_DIR/convert_hf_to_gguf.py"

# Remove the `llama.cpp` directory
cd "$CURRENT_DIR"
rm -rf "$FOLDER_NAME"

echo "Installation completed."
