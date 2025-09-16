#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Use this script to download datasets in your sandbox
set -x
sudo mkdir -m 777 -p $DATA_ROOT/datasets/lm_datasets
az login --use-device-code

# Phi-1 pre-training datasets
az storage copy -d "$DATA_ROOT/datasets/lm_datasets/the-stack-dedup-python-filtered/" -s "https://phyagi.blob.core.windows.net/data/datasets/lm_datasets/the-stack-dedup-python-filtered/5" --recursive --auth-mode login
az storage copy -d "$DATA_ROOT/datasets/lm_datasets/textbook/" -s "https://phyagi.blob.core.windows.net/data/datasets/lm_datasets/textbook/v4" --recursive --auth-mode login
az storage copy -d "$DATA_ROOT/datasets/lm_datasets/stackoverflow-with-meta-data-filtered/" -s "https://phyagi.blob.core.windows.net/data/datasets/lm_datasets/stackoverflow-with-meta-data-filtered/6" --recursive --auth-mode login
az storage copy -d "$DATA_ROOT/datasets/lm_datasets/code_contest/" -s "https://phyagi.blob.core.windows.net/data/datasets/lm_datasets/code_contest/CCa" --recursive --auth-mode login

# Phi-1 fine-tuning dataset
az storage copy -d "$DATA_ROOT/datasets/lm_datasets/baby-python/" -s "https://phyagi.blob.core.windows.net/data/datasets/lm_datasets/baby-python/v1" --recursive --auth-mode login

# Phi-1 evaluation datasets
az storage copy -d "$DATA_ROOT/datasets/lm_datasets/" -s "https://phyagi.blob.core.windows.net/data/datasets/lm_datasets/omg-08" --recursive --auth-mode login
az storage copy -d "$DATA_ROOT/datasets/lm_datasets/" -s "https://phyagi.blob.core.windows.net/data/datasets/openai_humaneval" --recursive --auth-mode login
az storage copy -d "$DATA_ROOT/datasets/lm_datasets/" -s "https://phyagi.blob.core.windows.net/data/datasets/p3/" --recursive --auth-mode login

# Large datasets
# az storage copy -d "$DATA_ROOT/datasets/lm_datasets/" -s "https://phyagi.blob.core.windows.net/data/datasets/webdata/" --recursive --auth-mode login
# az storage copy -d "$DATA_ROOT/datasets/lm_datasets/" -s "https://phyagi.blob.core.windows.net/data/datasets/stack-dedup-python/" --recursive --auth-mode login
# az storage copy -d "$DATA_ROOT/datasets/lm_datasets/" -s "https://phyagi.blob.core.windows.net/data/datasets/stackoverflow/" --recursive --auth-mode login

# Phi-4 reasoning seed prompts
# l2bcopy_args="--overwrite=ifSourceNewer --s2s-preserve-access-tier=false --check-length=true --include-directory-stub=false --s2s-preserve-blob-tags=false --recursive --log-level=INFO"
# azcopy copy https://aifshared.blob.core.windows.net/data/tool_use/data/phi4_reasoning_prompts/* "/data/sahagar/datasets/seeds/phi/phi4_reasoning_prompts/" $l2bcopy_args
# azcopy copy https://aifshared.blob.core.windows.net/data/tool_use/data/all_seeds/* "/data/sahagar/datasets/seeds/phi/all_seeds/" $l2bcopy_args