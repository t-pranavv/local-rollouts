#!/bin/bash

# Set defaults
DTYPE=${DTYPE:-bfloat16}
API_KEY=${API_KEY:-""}
SEED=${SEED:-42}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
TPP=$(( TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE ))
DATA_PARALLEL_SIZE_LOCAL=${DATA_PARALLEL_SIZE_LOCAL:-1}
GPUS=${GPU_PER_NODE_COUNT:-1}
BASE_PORT=${PORT:-9000}

# Build the base vllm command
base_cmd=(
    vllm serve "$MODEL_PATH"
    --tokenizer "$TOKENIZER_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --max-model-len "$MAX_MODEL_LEN"
    --dtype "$DTYPE"
    --seed "$SEED"
    --generation-config "$GENERATION_CONFIG"
    --override-generation-config "$OVERRIDE_GENERATION_CONFIG"
    --api-key "$API_KEY"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"
)

# Launch one server per local DDP rank
for ((local_dp_rank=0; local_dp_rank<DATA_PARALLEL_SIZE_LOCAL; local_dp_rank++)); do
    global_dp_rank=$(( NODE_RANK * DATA_PARALLEL_SIZE_LOCAL + local_dp_rank ))
    # compute a unique port
    serve_port=$(( BASE_PORT + local_dp_rank ))

    # build the full command
    cmd=( "${base_cmd[@]}" --port "$serve_port" )

    echo "Launching vllm server on local_dp_rank=$local_dp_rank (global_dp_rank=$global_dp_rank) on port $serve_port"

    # Compute the start and end GPU indices for this local DP rank
    start=$(( local_dp_rank * TPP ))
    end=$(( (local_dp_rank + 1) * TPP - 1 ))
    devices=$(seq -s, "$start" "$end")
    echo "Using GPUs: $devices"
    echo "Executing: ${cmd[@]}"

    # launch
    CUDA_VISIBLE_DEVICES=$devices "${cmd[@]}" > /dev/null &
    sleep 3
done