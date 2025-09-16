#!/bin/bash
# Script to start VLLM servers for inference

# General Configuration
export MODEL_NAME="Qwen3-32B"
export MODEL_PATH="Qwen/Qwen3-32B"
export TOKENIZER_PATH="Qwen/Qwen3-32B"
export SERVED_MODEL_NAME=$(echo vllm-local-$MODEL_NAME)
export MAX_MODEL_LEN=32768
export DTYPE="bfloat16"
export MODEL_UTILS="qwen_v1_msri_data"
export SYSMSG="re_tool_qwen_template_sys"
export CODE_INTERPRETER="LOCAL"
export PYBOX_RUNTIME="docker"
export PYBOX_MAX_SESSIONS=25
export PYBOX_COMMON_PIP_PACKAGES="NO INSTALL"
export PYBOX_BASE_IMAGE="aifrontiers.azurecr.io/python3.12-slim-pybox:160825"
export SEED=42
export GENERATION_CONFIG="auto"
export temperature=0.9
export OVERRIDE_GENERATION_CONFIG="{\"temperature\": $temperature, \"skip_special_tokens\": false}"
export JUDGE_GENERATION_CONFIG="{}"

# Experiment Configuration
export EXPERIMENT_NAME=eval_$(date +%Y-%m-%d_%H-%M-%S)
export EXPERIMENT_OUTPUT_DIR=$(echo outputs/$MODEL_NAME/$EXPERIMENT_NAME/seed_${SEED}_temperature_${temperature})
export VLLM_LOG_FILE=$EXPERIMENT_OUTPUT_DIR/vllm_logs.txt
export VLLM_MONITOR_LOG_FILE=$EXPERIMENT_OUTPUT_DIR/monitor_logs.txt

# GPU Configuration
export NODE_RANK=${NODE_RANK:-0}
export NODES=${NODES:-1}
export GPUS=${GPUS:-1}
export WORLD_SIZE=$(( NODES * GPUS ))
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-8100}

# Server Configuration
export API_KEY="key"
export PORT=9600
export TENSOR_PARALLEL_SIZE=1
export PIPELINE_PARALLEL_SIZE=1
export TPP=$(( TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE ))
export TOTAL_GPUS=$(( NODES * GPUS ))
export DATA_PARALLEL_SIZE=$(( TOTAL_GPUS / TPP > 0 ? TOTAL_GPUS / TPP : 1 ))
export DATA_PARALLEL_SIZE_LOCAL=$(( GPUS / TPP > 0 ? GPUS / TPP : 1 ))
export VLLM_DP_MASTER_IP=0.0.0.0
export API_SERVER_COUNT=$DATA_PARALLEL_SIZE

echo "Starting VLLM servers with the following configuration:"
echo "============================ GENERAL CONFIGURATION ============================"
echo "MODEL_NAME: $MODEL_NAME"
echo "MODEL_PATH: $MODEL_PATH"
echo "TOKENIZER_PATH: $TOKENIZER_PATH"
echo "SERVED_MODEL_NAME: $SERVED_MODEL_NAME"
echo "MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "DTYPE: $DTYPE"
echo "SEED: $SEED"
echo "GENERATION_CONFIG: $GENERATION_CONFIG"
echo "OVERRIDE_GENERATION_CONFIG: $OVERRIDE_GENERATION_CONFIG"
echo "JUDGE_GENERATION_CONFIG: $JUDGE_GENERATION_CONFIG"
echo "SYSMSG: $SYSMSG"
echo "============================ EXPERIMENT CONFIGURATION ============================"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
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
echo "============================ PYBOX RUNTIME CONFIGURATION ============================"
echo "PYBOX_RUNTIME: $PYBOX_RUNTIME"
echo "PYBOX_MAX_SESSIONS: $PYBOX_MAX_SESSIONS"
echo "PYBOX_BASE_IMAGE: $PYBOX_BASE_IMAGE"
echo "PYBOX_COMMON_PIP_PACKAGES: $PYBOX_COMMON_PIP_PACKAGES"
echo "==============================================================================="

# create the experiment output directory
mkdir -p $EXPERIMENT_OUTPUT_DIR

# Start VLLM servers
bash ../start_vllm_servers.sh

# # Login into MSR LIT and aifrontiers ACR for pybox docker image
# az account set --subscription "MSR LIT"
# az acr login -n aifrontiers -g aifrontiers

# get phyagi api key from azure key vault
export PHYAGI_API_KEY=$(python ../keyvault_secrets.py --secret-name phigen-api-key-eval --get --output_only_secret)

streamlit run app.py

if [[ $PYBOX_RUNTIME == "docker" ]]; then
    echo "Cleaning up Docker containers..."
    # Stop all running containers
    docker stop $(docker ps -q) || true
    # Remove all stopped containers (prune)
    docker container prune -f
else
    echo "PYBOX_RUNTIME is not set to docker, skipping Docker cleanup."
fi

# Kill all running VLLM servers
VLLM_PIDS=($(pgrep -f "vllm serve"))
if [[ ${#VLLM_PIDS[@]} -gt 0 ]]; then
    echo "Stopping VLLM servers..."
    pkill -f "vllm serve"
    pkill -f "VLLM"
else
    echo "No VLLM servers found to stop."
fi

exit 0