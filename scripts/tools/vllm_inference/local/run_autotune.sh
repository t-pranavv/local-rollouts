#!/bin/bash
# Script to Autotune VLLM servers for inference

# General Configuration
export MODEL_NAME="Phi-4-reasoning-hf"
export MODEL_PATH="microsoft/Phi-4-reasoning"
export TOKENIZER_PATH="microsoft/Phi-4-reasoning"
export MAX_MODEL_LEN=32768
export DTYPE="auto"
export SEED=42
export GENERATION_CONFIG="auto"
export temperature=0.8
export OVERRIDE_GENERATION_CONFIG="{\"temperature\": $temperature, \"skip_special_tokens\": false}"

# Experiment Configuration
export EXPERIMENT_OUTPUT_DIR=$(echo ../local/outputs/autotune/$MODEL_NAME/$EXPERIMENT_NAME/seed_${SEED}_temperature_${temperature})
export VLLM_LOG_FILE=$EXPERIMENT_OUTPUT_DIR/vllm_logs.txt
export VLLM_MONITOR_LOG_FILE=$EXPERIMENT_OUTPUT_DIR/monitor_logs.txt

# GPU Configuration
export NODE_RANK=${NODE_RANK:-0}
export NODES=${NODES:-1}
export GPUS=${GPUS:-4}
export WORLD_SIZE=$(( NODES * GPUS ))
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-8100}

# Server Configuration
export PORT=9600
export TENSOR_PARALLEL_SIZE=1
export PIPELINE_PARALLEL_SIZE=1
export TPP=$(( TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE ))
export TOTAL_GPUS=$(( NODES * GPUS ))
export DATA_PARALLEL_SIZE=$(( TOTAL_GPUS / TPP > 0 ? TOTAL_GPUS / TPP : 1 ))
export DATA_PARALLEL_SIZE_LOCAL=$(( GPUS / TPP > 0 ? GPUS / TPP : 1 ))
export VLLM_DP_MASTER_IP=0.0.0.0
export API_SERVER_COUNT=$DATA_PARALLEL_SIZE

# Auto-tuning Configuration
export MAX_INPUT_LEN=28672
export MAX_OUTPUT_LEN=4096
export NUM_SEQS_LIST="128 256 512"
export NUM_BATCHED_TOKENS_LIST="512 1024 2048 4096"

echo "Starting VLLM servers with the following configuration:"
echo "============================ GENERAL CONFIGURATION ============================"
echo "MODEL_NAME: $MODEL_NAME"
echo "MODEL_PATH: $MODEL_PATH"
echo "TOKENIZER_PATH: $TOKENIZER_PATH"
echo "MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "DTYPE: $DTYPE"
echo "SEED: $SEED"
echo "GENERATION_CONFIG: $GENERATION_CONFIG"
echo "OVERRIDE_GENERATION_CONFIG: $OVERRIDE_GENERATION_CONFIG"
echo "============================ EXPERIMENT CONFIGURATION ============================"
echo "EXPERIMENT_OUTPUT_DIR: $EXPERIMENT_OUTPUT_DIR"
echo "VLLM_LOG_FILE: $VLLM_LOG_FILE"
echo "VLLM_MONITOR_LOG_FILE: $VLLM_MONITOR_LOG_FILE"
echo "============================ GPU CONFIGURATION ============================"
echo "NODE_RANK: $NODE_RANK"
echo "NODES: $NODES"
echo "GPUS: $GPUS"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "============================ SERVER CONFIGURATION ============================"
echo "API_SERVER_COUNT: $API_SERVER_COUNT"
echo "PORT: $PORT"
echo "TENSOR_PARALLEL_SIZE: $TENSOR_PARALLEL_SIZE"
echo "PIPELINE_PARALLEL_SIZE: $PIPELINE_PARALLEL_SIZE"
echo "TPP: $TPP"
echo "TOTAL_GPUS: $TOTAL_GPUS"
echo "DATA_PARALLEL_SIZE: $DATA_PARALLEL_SIZE"
echo "DATA_PARALLEL_SIZE_LOCAL: $DATA_PARALLEL_SIZE_LOCAL"
echo "VLLM_DP_MASTER_IP: $VLLM_DP_MASTER_IP"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "============================ AUTO-TUNING CONFIGURATION ============================"
echo "MAX_INPUT_LEN: $MAX_INPUT_LEN"
echo "MAX_OUTPUT_LEN: $MAX_OUTPUT_LEN"
echo "NUM_SEQS_LIST: $NUM_SEQS_LIST"
echo "NUM_BATCHED_TOKENS_LIST: $NUM_BATCHED_TOKENS_LIST"
echo "==============================================================================="

# create the experiment output directory
mkdir -p $EXPERIMENT_OUTPUT_DIR

cd ../modeltune/
bash autotune.sh

# Kill all running VLLM servers
VLLM_PIDS=($(pgrep -f "vllm serve"))
if [[ ${#VLLM_PIDS[@]} -gt 0 ]]; then
    echo "Stopping VLLM servers..."
    pkill -f "vllm serve"
else
    echo "No VLLM servers found to stop."
fi