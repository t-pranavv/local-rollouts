#!/bin/bash

# Default values
WORLD_SIZE=""
PRETRAINED_MODEL=""
TOKENIZER_MODEL=""
BASE_PORT=""
API_KEY=""
MAX_MODEL_LEN=""
SERVED_MODEL_NAME=""
DTYPE="bfloat16"
SEED=""
MAX_NUM_SEQS="128"
GENERATION_CONFIG=""
OVERRIDE_GENERATION_CONFIG=""


# Function to display usage
usage() {
    echo "Usage: $0 --world-size=<num> --pretrained-model=<path> --tokenizer-model=<path> --base-port=<num> --api-key=<key> --max-model-len=<num> --served-model-name=<name> --dtype=<dtype> --seed=<num> --max-num-seqs=<num> --generation-config=<path> --override-generation-config=<path>"
    exit 1
}

# Parse command-line arguments
for arg in "$@"; do
    case $arg in
        --world-size=*) WORLD_SIZE="${arg#*=}" ;;
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

# Ensure all required arguments are provided
if [[ -z "$WORLD_SIZE" || -z "$PRETRAINED_MODEL" || -z "$TOKENIZER_MODEL" || -z "$BASE_PORT" || -z "$API_KEY" || -z "$MAX_MODEL_LEN" || -z "$SERVED_MODEL_NAME" || -z "$DTYPE" || -z "$SEED" || -z "$MAX_NUM_SEQS" || -z "$GENERATION_CONFIG" || -z "$OVERRIDE_GENERATION_CONFIG" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# Loop through each rank
for (( GPURANK=0; GPURANK<WORLD_SIZE; GPURANK++ )); do
    # Set CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES=$GPURANK

    # Compute the port number
    PORT=$((BASE_PORT + GPURANK))

    # Log the command before execution
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

    # Run the command silently
    eval $CMD > /dev/null &
    sleep 3
done