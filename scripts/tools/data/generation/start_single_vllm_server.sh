#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Default values
PRETRAINED_MODEL=""
TOKENIZER_MODEL=""
BASE_PORT=""
API_KEY=""
MAX_MODEL_LEN=""
SERVED_MODEL_NAME=""
DTYPE=""
SEED=""
MAX_NUM_SEQS="128"
GENERATION_CONFIG=""
OVERRIDE_GENERATION_CONFIG=""

# Function to display usage
usage() {
    echo "Usage: $0 --pretrained-model=<path> --tokenizer-model=<path> --base-port=<num> --api-key=<key> --max-model-len=<num> --served-model-name=<name> --dtype=<dtype> --seed=<num> --max-num-seqs=<num> --generation-config=<path> --override-generation-config=<path>"
    exit 1
}

# Parse command-line arguments
for arg in "$@"; do
    case $arg in
        --pretrained-model=*) PRETRAINED_MODEL="${arg#*=}" ;;
        --tokenizer-model=*) TOKENIZER_MODEL="${arg#*=}" ;;
        --base-port=*) BASE_PORT="${arg#*=}" ;;
        --api-key=*) API_KEY="${arg#*=}" ;;
        --max-model-len=*) MAX_MODEL_LEN="${arg#*=}" ;;
        --served-model-name=*) SERVED_MODEL_NAME="${arg#*=}" ;;
        --dtype=*) DTYPE="${arg#*=}" ;;
        --seed=*) SEED="${arg#*=}" ;;
        --max-num-seqs=*) MAX_NUM_SEQS="${arg#*=}" ;;
        --generation-config=*) GENERATION_CONFIG="${arg#*=}" ;;
        --override-generation-config=*) OVERRIDE_GENERATION_CONFIG="${arg#*=}" ;;
        *) echo "Unknown argument: $arg"; usage ;;
    esac
done

echo "PRETRAINED_MODEL: $PRETRAINED_MODEL
TOKENIZER_MODEL: $TOKENIZER_MODEL
BASE_PORT: $BASE_PORT
API_KEY: $API_KEY
MAX_MODEL_LEN: $MAX_MODEL_LEN
SERVED_MODEL_NAME: $SERVED_MODEL_NAME
DTYPE: $DTYPE
SEED: $SEED
MAX_NUM_SEQS: $MAX_NUM_SEQS
GENERATION_CONFIG: $GENERATION_CONFIG
OVERRIDE_GENERATION_CONFIG: $OVERRIDE_GENERATION_CONFIG"

# Ensure all required arguments are provided
if [[ -z "$RANK" || -z "$PRETRAINED_MODEL" || -z "$TOKENIZER_MODEL" || -z "$BASE_PORT" || -z "$API_KEY" || -z "$MAX_MODEL_LEN" || -z "$SERVED_MODEL_NAME" || -z "$DTYPE" || -z "$SEED" || -z "$MAX_NUM_SEQS" || -z "$GENERATION_CONFIG" || -z "$OVERRIDE_GENERATION_CONFIG" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# Set CUDA_VISIBLE_DEVICES for the given rank
export CUDA_VISIBLE_DEVICES=$RANK

# Compute the port number for the given rank
PORT=$((BASE_PORT + RANK))

# Log the command to be executed
CMD="vllm serve \"$PRETRAINED_MODEL\" \
        --tokenizer \"$TOKENIZER_MODEL\" \
        --port \"$PORT\" \
        --api-key \"$API_KEY\" \
        --max-model-len \"$MAX_MODEL_LEN\" \
        --served-model-name \"$SERVED_MODEL_NAME\" \
        --dtype \"$DTYPE\" \
        --seed \"$SEED\" \
        --max-num-seqs \"$MAX_NUM_SEQS\" \
        --generation-config \"$GENERATION_CONFIG\" \
        --override-generation-config \"$OVERRIDE_GENERATION_CONFIG\""

echo "Executing: $CMD"

# Run the VLLM server command for the current rank silently
eval $CMD > /dev/null &