#!/bin/bash
# Script to start VLLM servers for inference

# General Configuration
export MODEL_NAME="Qwen3-32B"
export MODEL_UTILS="qwen_v1_msri_data"
export SYSMSG="re_tool_qwen_template_sys"
export SEED=42
export temperature=0.9

# Experiment Configuration
export EXPERIMENT_NAME=eval_2025-08-26_08-06-56
export EXPERIMENT_OUTPUT_DIR=$(echo outputs/$MODEL_NAME/$EXPERIMENT_NAME/seed_${SEED}_temperature_${temperature})
export OUTPUT_FILENAME="tool_use"
export LOCAL_DB_DIR=$(echo $EXPERIMENT_OUTPUT_DIR/$OUTPUT_FILENAME/)
export LOCAL_DB_NAME="${OUTPUT_FILENAME}_output_dataset"

echo "============================ GENERAL CONFIGURATION ============================"
echo "MODEL_NAME: $MODEL_NAME"
echo "SYSMSG: $SYSMSG"
echo "SEED: $SEED"
echo "TEMPERATURE: $temperature"
echo "============================ EXPERIMENT CONFIGURATION ============================"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "EXPERIMENT_OUTPUT_DIR: $EXPERIMENT_OUTPUT_DIR"

# get phyagi api key from azure key vault
export PHYAGI_API_KEY=$(python ../keyvault_secrets.py --secret-name phigen-api-key-eval --get --output_only_secret)

benchmarks=("$OUTPUT_FILENAME")
for i in "${!benchmarks[@]}"; do
    benchmark=${benchmarks[i]}
    export PROMPT_COLUMN="prompt"

    export OUTDIR=$(echo $EXPERIMENT_OUTPUT_DIR/$benchmark/)
    echo "EVAL_OUTDIR: $OUTDIR"

    python write_partial_generations.py  \
        --local_db_dir $LOCAL_DB_DIR \
        --local_db_name $LOCAL_DB_NAME \
        --prompts_file ${benchmark}.jsonl \
        --prompt_field $PROMPT_COLUMN \
        --output_dir $OUTDIR \
        --agent_cls "VLLMLocalToolResponseAgent" \
        --api_type "completion" \
        --model_utils_name $MODEL_UTILS \
        --system_message $SYSMSG \
        --thinking_model
done