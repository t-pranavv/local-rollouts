# Tools

This directory contains various helper scripts and utilities for data processing, installation of dependencies, distributed computing tests, and environment-specific tasks such as container log handling and model quantization.

## Data

Contains all data-related scripts and files, organized into the following subfolders:

* `generation`: Scripts for text generation tasks, possibly used for evaluating models or preparing outputs.
* `preprocessing`: Utilities for cleaning, formatting, and preparing raw data before training or inference.
* `tokenization`: Tools for applying or testing different tokenization strategies.

## Installation

Shell scripts for setting up system-level or package-level dependencies:

* `flash-attn.sh`: Installs [`flash-attn`](https://github.com/Dao-AILab/flash-attention), a fast attention implementation for transformer models.
* `llama-cpp.sh`: Installs [`llama.cpp`](https://github.com/ggerganov/llama.cpp), a C++ implementation of LLaMA inference.
* `miniconda.sh`: Installs [`miniconda`](https://www.anaconda.com/docs/getting-started/miniconda/main), a miniature installation of Anaconda distribution.

## Singularity

Scripts related to containerized environments and distributed computation testing:

* `all_reduce_test.py`: A diagnostic script to verify performance and correctness of collective communication (e.g., NCCL or MPI-based all-reduce).
* `download_singularity_logs.py`: Automates the retrieval of logs from Singularity containers for monitoring or debugging.

## Additional scripts

Standalone or general-purpose scripts outside the categorized folders, such as:

* Quantization tools (e.g., `convert_and_quantize_to_gguf.sh`).
* Experimental utilities.
