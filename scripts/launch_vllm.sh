#!/bin/bash
set -e

# -----------------------------
# Configurable variables
# -----------------------------
MODEL_PATH="microsoft/phi-4"
TP_SIZE=1
SERVER_PORT=8000
export MODEL_NAME="phi-4-hf"

# -----------------------------
# Start vLLM server
# -----------------------------
echo "Launching vLLM server for $MODEL_PATH ..."
vllm serve "$MODEL_PATH" \
  --tensor-parallel-size $TP_SIZE \
  --served-model-name "vllm-local-$MODEL_NAME" \
  --port $SERVER_PORT &
SERVER_PID=$!