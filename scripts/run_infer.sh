export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "compat/lib" | tr '\n' ':' | sed 's/:$//')

# Print IB topology
nvidia-smi topo -m

# General Configuration
# export MODEL_NAME=$(echo ${{search_space.model_config}} | cut -d',' -f1)
# export MODEL_PATH=$(echo ${{search_space.model_config}} | cut -d',' -f2)
# export TOKENIZER_PATH=$(echo ${{search_space.model_config}} | cut -d',' -f3)
# export SERVED_MODEL_NAME=$(echo vllm-local-$MODEL_NAME)
# export MAX_MODEL_LEN=$(echo ${{search_space.model_config}} | cut -d',' -f4)
# export DTYPE=$(echo ${{search_space.model_config}} | cut -d',' -f5)
# export MODEL_UTILS=$(echo ${{search_space.model_config}} | cut -d',' -f8)
# export SYSMSG=$(echo ${{search_space.model_config}} | cut -d',' -f9)

export MODEL_NAME="phi-4-hf"
export MODEL_PATH="microsoft/phi-4"
export TOKENIZER_PATH="microsoft/phi-4"
export SERVED_MODEL_NAME="vllm-local-$MODEL_NAME"
export MAX_MODEL_LEN="32768"
export DTYPE="bfloat16"
export MODEL_UTILS="phi4_v1_tent_data"
export SYSMSG="re_tool_qwen_template_sys"


export PYBOX_DOCKER_ARCHIVE="${{outputs.local_docker_images}}/python3.10-slim-pybox.tar"
export PYBOX_APPTAINER_IMAGE="${{outputs.local_docker_images}}/python3.10-slim-pybox.sif"
export PYBOX_WHEELS_DIR="${{inputs.ci_installation_wheels}}"
export SEED=${{search_space.seed}}
export GENERATION_CONFIG="auto"
export OVERRIDE_GENERATION_CONFIG='{"temperature": ${{search_space.temperature}}, "skip_special_tokens": false}'
export JUDGE_GENERATION_CONFIG="{}"
export JUDGE_MODEL_NAME="gpt-4o-impact"

# Experiment Configuration
export EXPERIMENT_NAME=tooluse_eval_$(date +%Y-%m-%d_%H-%M-%S)
export EXPERIMENT_OUTPUT_DIR=$(echo ${{outputs.output_dir}}/$MODEL_NAME/$EXPERIMENT_NAME/seed_${SEED}_temperature_${{search_space.temperature}})

# GPU Configuration
export NODE_RANK=${NODE_RANK:-0}
export NODES=${AZUREML_NODE_COUNT:-1}
export GPUS=${GPU_PER_NODE_COUNT:-1}
export WORLD_SIZE=$(( NODES * GPUS ))

# Server Configuration
export API_KEY="key"
export PORT=8000
export TENSOR_PARALLEL_SIZE=$(echo ${{search_space.model_config}} | cut -d',' -f6)
export PIPELINE_PARALLEL_SIZE=$(echo ${{search_space.model_config}} | cut -d',' -f7)
export TPP=$(( TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE ))
export TOTAL_GPUS=$(( NODES * GPUS ))
export DATA_PARALLEL_SIZE=$(( TOTAL_GPUS / TPP > 0 ? TOTAL_GPUS / TPP : 1 ))
export DATA_PARALLEL_SIZE_LOCAL=$(( GPUS / TPP > 0 ? GPUS / TPP : 1 ))
export VLLM_DP_MASTER_IP=0.0.0.0
export API_SERVER_COUNT=$DATA_PARALLEL_SIZE

# create the experiment output directory
mkdir -p $EXPERIMENT_OUTPUT_DIR

# cd into vllm inference scripts directory
cd tools/vllm_inference

# get phyagi api key from azure key vault
export PHYAGI_API_KEY=91de1961565647aba611fdb17222b084

# model checkpoint conversion
if [[ $MODEL_NAME == *-hf ]]; then
      echo "Skip conversion due to -hf suffix indicating HF model..."
      if [[ -d ${{outputs.output_ckpt}}/huggingface_models/$MODEL_PATH ]]; then
        echo "HF model already exists..."
      else
        echo "Downloading HF model from Hugging Face..."
        if [ $NODE_RANK -eq 0 ]; then
          python download_and_save_hf_models.py -m $MODEL_PATH -t $TOKENIZER_PATH -o ${{outputs.output_ckpt}}/huggingface_models
          echo "HF model downloaded and saved to ${{outputs.output_ckpt}}/huggingface_models/$MODEL_PATH"
        else
          echo "Node $NODE_RANK waiting for HF model to be downloaded on rank 0"
          while [ ! -d "${{outputs.output_ckpt}}/huggingface_models/$MODEL_PATH" ]; do
            sleep 5
          done
          echo "HF model downloaded, proceeding on node $NODE_RANK..."
        fi
      fi
      export MODEL_PATH=${{outputs.output_ckpt}}/huggingface_models/$MODEL_PATH
      export TOKENIZER_PATH=$MODEL_PATH
elif [[ -d ${{outputs.output_ckpt}}/$MODEL_PATH/phi4 ]]; then
      echo "Skip conversion, hf in phi4 class model already exists..."
      export MODEL_PATH=${{outputs.output_ckpt}}/$MODEL_PATH/phi4
      export TOKENIZER_PATH=${{inputs.tokenizer}}/$TOKENIZER_PATH
else
      echo "Converting model from Zero3 to FP32 and then to HF format..."
      if [ $NODE_RANK -eq 0 ]; then
        echo "Node $NODE_RANK starting conversion..."
        python -m phyagi.cli.interface convert ${{outputs.output_ckpt}}/$MODEL_PATH phi4 -t ${{inputs.tokenizer}}/$TOKENIZER_PATH --dtype=bfloat16 --debug_logit --debug_params --from_deepspeed_zero --save_intermediate_checkpoint # (for saving convertion to fp32)
        echo "Conversion completed on rank 0"
      else
        echo "Node $NODE_RANK waiting for conversion to complete on rank 0..."
        while [ ! -d "${{outputs.output_ckpt}}/$MODEL_PATH/phi4" ]; do
          sleep 5
        done
        echo "Conversion completed, proceeding on node $NODE_RANK..."
      fi
      export MODEL_PATH=${{outputs.output_ckpt}}/$MODEL_PATH/phi4
      export TOKENIZER_PATH=${{inputs.tokenizer}}/$TOKENIZER_PATH
fi

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
echo "PYBOX_DOCKER_ARCHIVE: $PYBOX_DOCKER_ARCHIVE"
echo "PYBOX_APPTAINER_IMAGE: $PYBOX_APPTAINER_IMAGE"
echo "PYBOX_WHEELS_DIR: $PYBOX_WHEELS_DIR"
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
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "============================ PYBOX RUNTIME CONFIGURATION ============================"
echo "CODE_INTERPRETER: $CODE_INTERPRETER"
echo "CODE_INTERPRETER_MI_CLIENT_ID: $CODE_INTERPRETER_MI_CLIENT_ID"
echo "PYBOX_RUNTIME: $PYBOX_RUNTIME"
echo "PYBOX_MAX_SESSIONS: $PYBOX_MAX_SESSIONS"
echo "PYBOX_APPTAINER_IMAGE: $PYBOX_APPTAINER_IMAGE"
echo "PYBOX_DOCKER_ARCHIVE: $PYBOX_DOCKER_ARCHIVE"
echo "PYBOX_WHEELS_DIR: $PYBOX_WHEELS_DIR"
echo "APPTAINER_FALLBACK_MODE: $APPTAINER_FALLBACK_MODE"
echo "PYBOX_COMMON_PIP_PACKAGES: $PYBOX_COMMON_PIP_PACKAGES"
echo "PYBOX_OFFLINE: $PYBOX_OFFLINE"
echo "==============================================================================="

    # Start VLLM servers
    # bash start_vllm_servers.sh

if [ $NODE_RANK -eq 0 ]; then
      benchmarks=("aime_2024" "gpqa_diamond")
      EVAL_TYPES=("AIME_SIMPLE AIME" "GPQA")
      for i in "${!benchmarks[@]}"; do
        benchmark=${benchmarks[i]}
        if [[ $benchmark == aime_202* ]]; then
          export PROMPT_COLUMN="problem" 
          export NUM_SAMPLES_TO_GENERATE=5
        else
          export PROMPT_COLUMN="prompt"
          export NUM_SAMPLES_TO_GENERATE=5
        fi

        export OUTDIR=$(echo $EXPERIMENT_OUTPUT_DIR/$benchmark/)
        rm -rf $OUTDIR/*.db*
        mkdir -p $OUTDIR
        echo "EVAL_OUTDIR: $OUTDIR"
        
        python run_inference_on_vllm_server.py  \
          --base_ip $VLLM_DP_MASTER_IP \
          --base_port $PORT \
          --num_servers $API_SERVER_COUNT \
          --served_model_name $SERVED_MODEL_NAME \
          --prompts_file ${{inputs.simpleevalprompts}}/${benchmark}.jsonl \
          --prompt_field $PROMPT_COLUMN \
          --output_dir $OUTDIR \
          --tokenizer_path $TOKENIZER_PATH \
          --agent_cls "VLLMLocalToolResponseAgent" \
          --api_type "completion" \
          --max_tool_call_steps 5 \
          --tool_call_timeouts '{"code_interpreter": {"python": 200}}' \
          --model_utils_name $MODEL_UTILS \
          --max_model_seq_len $MAX_MODEL_LEN \
          --system_message $SYSMSG \
          --thinking_model \
          --num_worker_procs $TOTAL_GPUS \
          --num_samples_to_generate $NUM_SAMPLES_TO_GENERATE \
          --generation_config "$OVERRIDE_GENERATION_CONFIG"

        # run gpt judge
        # IFS=' ' read -r -a eval_types <<< "${EVAL_TYPES[i]}"
        # for eval_type in "${eval_types[@]}"; do
        #   export RESULTS_DIR=$OUTDIR
        #   export JUDGE_OUTDIR=$OUTDIR/$eval_type/
        #   rm -rf $JUDGE_OUTDIR/*.db*
        #   mkdir -p $JUDGE_OUTDIR
        #   echo "RESULTS_DIR: $RESULTS_DIR"
        #   echo "JUDGE_OUTDIR: $JUDGE_OUTDIR"

        #   python run_gpt_judge.py \
        #     --prompts_file $RESULTS_DIR/${benchmark}_output.jsonl \
        #     --prompt_field $PROMPT_COLUMN \
        #     --answer_field "answer" \
        #     --completions_field "completions" \
        #     --completion_type "response" \
        #     --model_utils_name $MODEL_UTILS \
        #     --tool_call_timeouts '{"code_interpreter": {"python": 200}}' \
        #     --n_samples $NUM_SAMPLES_TO_GENERATE \
        #     --repeat 1 \
        #     --output_dir $JUDGE_OUTDIR \
        #     --judge_model_name $JUDGE_MODEL_NAME \
        #     --eval_type $eval_type \
        #     --num_worker_procs 512 \
        #     --generation_config "$JUDGE_GENERATION_CONFIG"
        # done
      done

    python summarize_results.py --output_dir $EXPERIMENT_OUTPUT_DIR
fi