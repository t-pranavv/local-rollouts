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
SEED="42"
MAX_NUM_SEQS="128"
GENERATION_CONFIG=""
OVERRIDE_GENERATION_CONFIG=""
TENSOR_PARALLEL_SIZE="1"


# Function to display usage
usage() {
    echo "Usage: $0 --world-size=<num> --pretrained-model=<path> --tokenizer-model=<path> --base-port=<num> --api-key=<key> --max-model-len=<num> --served-model-name=<name> --generation-config=<path> --override-generation-config=<path> --tensor-parallel-size=<num> [--dtype=<dtype>] [--seed=<num>] [--max-num-seqs=<num>] [--tensor-parallel-size=<num>] "
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
        --tensor-parallel-size=*) TENSOR_PARALLEL_SIZE="${arg#*=}" ;;
        --generation-config=*) GENERATION_CONFIG="${arg#*=}" ;;
        --override-generation-config=*) OVERRIDE_GENERATION_CONFIG="${arg#*=}" ;;
        *) echo "Unknown argument: $arg"; usage ;;
    esac
done

# Ensure all required arguments are provided
if [[ -z "$WORLD_SIZE" || -z "$PRETRAINED_MODEL" || -z "$TOKENIZER_MODEL" || -z "$BASE_PORT" || -z "$API_KEY" || -z "$MAX_MODEL_LEN" || -z "$SERVED_MODEL_NAME" || -z "$GENERATION_CONFIG" || -z "$OVERRIDE_GENERATION_CONFIG" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# Calculate number of servers to start
NUM_SERVERS=$((WORLD_SIZE / TENSOR_PARALLEL_SIZE))
echo "Starting $NUM_SERVERS server instances with tensor parallel size $TENSOR_PARALLEL_SIZE"

# Loop through each server instance
for (( SERVER=0; SERVER<NUM_SERVERS; SERVER++ )); do
    # Calculate GPU range for this server
    START_GPU=$((SERVER * TENSOR_PARALLEL_SIZE))
    END_GPU=$((START_GPU + TENSOR_PARALLEL_SIZE - 1))
    
    # Set CUDA_VISIBLE_DEVICES for tensor parallel GPUs
    GPU_LIST=""
    for (( GPU=START_GPU; GPU<=END_GPU; GPU++ )); do
        if [[ -z "$GPU_LIST" ]]; then
            GPU_LIST="$GPU"
        else
            GPU_LIST="$GPU_LIST,$GPU"
        fi
    done
    export CUDA_VISIBLE_DEVICES=$GPU_LIST
    export VLLM_DISABLE_COMPILE_CACHE=1    

    # Compute the port number
    PORT=$((BASE_PORT + SERVER))
    if [[ $END_GPU -ge 8 ]]; then
        echo "Error: GPU range $START_GPU-$END_GPU exceeds available GPUs (0-7)"
        exit 1
    fi

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
        --override-generation-config \"$OVERRIDE_GENERATION_CONFIG\" \
        --tensor-parallel-size \"$TENSOR_PARALLEL_SIZE\""

    echo "Starting server $SERVER with GPU_ids [$GPU_LIST] on port $PORT"    
    echo "Executing: $CMD"

    # Run the command silently
    eval $CMD > /dev/null &
    sleep 3
done