#!/usr/bin/env bash
set -euo pipefail

# Clean up LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "compat/lib" | tr '\n' ':' | sed 's/:$//')

# Print IB topology
nvidia-smi topo -m

# ============================ GENERAL CONFIGURATION ============================
export MODEL_NAME="phi-4-hf"
export MODEL_PATH="microsoft/phi-4"
export TOKENIZER_PATH="microsoft/phi-4"
export SERVED_MODEL_NAME="vllm-local-$MODEL_NAME"
export MAX_MODEL_LEN="32768"
export DTYPE="bfloat16"
export MODEL_UTILS="phi4_v1_tent_data"
export SYSMSG="re_tool_qwen_template_sys"

# export PYBOX_DOCKER_ARCHIVE="$LOCAL_DOCKER_IMAGES/python3.10-slim-pybox.tar"
# export PYBOX_APPTAINER_IMAGE="$LOCAL_DOCKER_IMAGES/python3.10-slim-pybox.sif"
# export PYBOX_WHEELS_DIR="$CI_INSTALLATION_WHEELS"
export SEED=42
export SEARCH_SPACE_TEMPERATURE=0.8
export GENERATION_CONFIG="auto"
export OVERRIDE_GENERATION_CONFIG="{\"temperature\": $SEARCH_SPACE_TEMPERATURE, \"skip_special_tokens\": false}"
export JUDGE_GENERATION_CONFIG="{}"
export JUDGE_MODEL_NAME="gpt-4o-impact"
export OUTPUT_DIR="/home/t-pranavv/phyagi-sdk/outputs"
# ============================ EXPERIMENT CONFIGURATION ============================
export EXPERIMENT_NAME=tooluse_eval_$(date +%Y-%m-%d_%H-%M-%S)
export EXPERIMENT_OUTPUT_DIR="$OUTPUT_DIR/$MODEL_NAME/$EXPERIMENT_NAME/seed_${SEED}_temperature_${SEARCH_SPACE_TEMPERATURE}"

# ============================ GPU CONFIGURATION ============================
export NODE_RANK=${NODE_RANK:-0}
export NODES=${AZUREML_NODE_COUNT:-1}
export GPUS=${GPU_PER_NODE_COUNT:-1}
export WORLD_SIZE=$(( NODES * GPUS ))

# ============================ SERVER CONFIGURATION ============================
export API_KEY="key"
export PORT=8000
export TENSOR_PARALLEL_SIZE=$(echo $SEARCH_SPACE_MODEL_CONFIG | cut -d',' -f6)
export PIPELINE_PARALLEL_SIZE=$(echo $SEARCH_SPACE_MODEL_CONFIG | cut -d',' -f7)
export TPP=$(( TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE ))
export TOTAL_GPUS=$(( NODES * GPUS ))

if [ $TPP -gt 0 ]; then
    export DATA_PARALLEL_SIZE=$(( TOTAL_GPUS / TPP ))
    export DATA_PARALLEL_SIZE_LOCAL=$(( GPUS / TPP ))
else
    export DATA_PARALLEL_SIZE=1
    export DATA_PARALLEL_SIZE_LOCAL=1
fi

export VLLM_DP_MASTER_IP=0.0.0.0
export API_SERVER_COUNT=$DATA_PARALLEL_SIZE

# Create experiment output directory
mkdir -p "$EXPERIMENT_OUTPUT_DIR"

# cd into vllm inference scripts directory
cd tools/vllm_inference

# ============================ API KEYS ============================
export PHYAGI_API_KEY="91de1961565647aba611fdb17222b084"
export OUTPUT_CKPT="$OUTPUT_DIR/checkpoints"

# ============================ MODEL CONVERSION ============================
if [[ $MODEL_NAME == *-hf ]]; then
    echo "Skip conversion due to -hf suffix indicating HF model..."
    if [[ -d $OUTPUT_CKPT/huggingface_models/$MODEL_PATH ]]; then
        echo "HF model already exists..."
    else
        echo "Downloading HF model from Hugging Face..."
        if [ $NODE_RANK -eq 0 ]; then
            python download_and_save_hf_models.py -m $MODEL_PATH -t $TOKENIZER_PATH -o $OUTPUT_CKPT/huggingface_models
            echo "HF model downloaded and saved to $OUTPUT_CKPT/huggingface_models/$MODEL_PATH"
        else
            echo "Node $NODE_RANK waiting for HF model to be downloaded on rank 0"
            while [ ! -d "$OUTPUT_CKPT/huggingface_models/$MODEL_PATH" ]; do
                sleep 5
            done
            echo "HF model downloaded, proceeding on node $NODE_RANK..."
        fi
    fi
    export MODEL_PATH=$OUTPUT_CKPT/huggingface_models/$MODEL_PATH
    export TOKENIZER_PATH=$MODEL_PATH
elif [[ -d $OUTPUT_CKPT/$MODEL_PATH/phi4 ]]; then
    echo "Skip conversion, hf in phi4 class model already exists..."
    export MODEL_PATH=$OUTPUT_CKPT/$MODEL_PATH/phi4
    export TOKENIZER_PATH=$INPUTS_TOKENIZER/$TOKENIZER_PATH
else
    echo "Converting model from Zero3 to FP32 and then to HF format..."
    if [ $NODE_RANK -eq 0 ]; then
        echo "Node $NODE_RANK starting conversion..."
        python -m phyagi.cli.interface convert $OUTPUT_CKPT/$MODEL_PATH phi4 -t $INPUTS_TOKENIZER/$TOKENIZER_PATH --dtype=bfloat16 --debug_logit --debug_params --from_deepspeed_zero --save_intermediate_checkpoint
        echo "Conversion completed on rank 0"
    else
        echo "Node $NODE_RANK waiting for conversion to complete on rank 0..."
        while [ ! -d "$OUTPUT_CKPT/$MODEL_PATH/phi4" ]; do
            sleep 5
        done
        echo "Conversion completed, proceeding on node $NODE_RANK..."
    fi
    export MODEL_PATH=$OUTPUT_CKPT/$MODEL_PATH/phi4
    export TOKENIZER_PATH=$INPUTS_TOKENIZER/$TOKENIZER_PATH
fi

# ============================ PRINT CONFIG ============================
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
echo "JUDGE_MODEL_NAME: $JUDGE_MODEL_NAME"
echo "SYSMSG: $SYSMSG"
# echo "PYBOX_DOCKER_ARCHIVE: $PYBOX_DOCKER_ARCHIVE"
# echo "PYBOX_APPTAINER_IMAGE: $PYBOX_APPTAINER_IMAGE"
# echo "PYBOX_WHEELS_DIR: $PYBOX_WHEELS_DIR"
echo "============================ EXPERIMENT CONFIGURATION ============================"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "EXPERIMENT_OUTPUT_DIR: $EXPERIMENT_OUTPUT_DIR"
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
echo "MASTER_ADDR: ${MASTER_ADDR:-unset}"
echo "MASTER_PORT: ${MASTER_PORT:-unset}"
echo "==============================================================================="

    # Start VLLM servers
# bash tools/vllm_inference/start_vllm_servers.sh

# ============================ RUN BENCHMARKS ============================
if [ $NODE_RANK -eq 0 ]; then
    benchmarks=("gpqa_diamond")
    for benchmark in "${benchmarks[@]}"; do
        if [[ $benchmark == aime_202* ]]; then
            export PROMPT_COLUMN="question"
            # export PROMPT_COLUMN="problem"
            export NUM_SAMPLES_TO_GENERATE=5
        else
            # export PROMPT_COLUMN="prompt"
            export PROMPT_COLUMN="question"
            export NUM_SAMPLES_TO_GENERATE=5
        fi

        export OUTDIR="$EXPERIMENT_OUTPUT_DIR/$benchmark/"
        rm -rf "$OUTDIR"/*.db*
        mkdir -p "$OUTDIR"
        echo "EVAL_OUTDIR: $OUTDIR"

        python run_inference_on_vllm_server.py \
            --base_ip $VLLM_DP_MASTER_IP \
            --base_port $PORT \
            --num_servers $API_SERVER_COUNT \
            --served_model_name $SERVED_MODEL_NAME \
            --prompts_file /home/t-pranavv/phyagi-sdk/prompts/aime_2025-AIME2025-I.jsonl \
            --prompt_field $PROMPT_COLUMN \
            --output_dir $OUTDIR \
            --tokenizer_path $TOKENIZER_PATH \
            --agent_cls "VLLMLocalToolResponseAgent" \
            --api_type "completion" \
            --max_tool_call_steps 5 \
            --tool_call_timeouts '{"code_interpreter": {"python": 5}}' \
            --model_utils_name $MODEL_UTILS \
            --max_model_seq_len $MAX_MODEL_LEN \
            --system_message $SYSMSG \
            --thinking_model \
            --num_worker_procs $TOTAL_GPUS \
            --num_samples_to_generate $NUM_SAMPLES_TO_GENERATE \
            --generation_config "$OVERRIDE_GENERATION_CONFIG"
    done

    python summarize_results.py --output_dir "$EXPERIMENT_OUTPUT_DIR"
fi
