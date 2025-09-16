#!/bin/bash

install_miniconda () {
    PREFIX=$1
    if [ -z "$PREFIX" ]; then
        echo "No prefix provided. Exiting."
    else
        mkdir -p "$PREFIX"
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$PREFIX/miniconda.sh"
        bash "$PREFIX/miniconda.sh" -b -u -p "$PREFIX"
        rm -rf "$PREFIX/miniconda.sh"
        "$PREFIX/bin/conda" init bash
        "$PREFIX/bin/conda" init zsh
    fi   
}

# Call the function with the provided prefix
install_miniconda "$1"