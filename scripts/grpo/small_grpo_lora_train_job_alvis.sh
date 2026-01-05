#!/bin/bash
#SBATCH -A NAISS2025-5-243
#SBATCH -p alvis
#SBATCH --job-name=crrl-small-grpo-lora
#SBATCH --output=logs/small_grpo_lora_%j.out
#SBATCH --error=logs/small_grpo_lora_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100fat:2
#SBATCH --time=24:00:00


# Small GRPO train job, 2 fat GPUs, 1 running vLLM, 1 training

# Apptainer common runtime configuration (requires CRRL_WORKDIR)
# source scripts/appt_common.sh

module load CUDA/13.0.2
export APPTAINER_CACHEDIR="/mimer/NOBACKUP/groups/naiss2025-5-243/andre/CodeRepairRL/.apptainer-cache"
export APPTAINER_TMPDIR=$TMPDIR

# MODEL_CONFIG can be provided via env or as --model_config <name>
MODEL_CONFIG="${MODEL_CONFIG:-small_qwen}"
if [[ "${1:-}" == --model_config=* ]]; then MODEL_CONFIG="${1#*=}"; shift; fi
if [[ "${1:-}" == --model_config ]]; then MODEL_CONFIG="${2:?}"; shift 2; fi


MASTER_PORT=43001
MODEL_NAME=$(awk -F '"' '/^model_name:/ {print $2; exit}' "src/conf/model/${MODEL_CONFIG}.yaml")

# Minimal parser selection based on model name and optional chat template/plugin
RP=""; TP=""; CT=""
case "${MODEL_CONFIG,,}" in
  *qwen*)     RP="--reasoning_parser qwen3"; TP="--tool_call_parser hermes";;
  *nemotron*) TP="--tool_call_parser llama3_json"; CT="--chat-template src/chat_templates/tool_chat_template_llama3.1_json.jinja";;
  *llama*)    TP="--tool_call_parser llama3_json"; CT="--chat-template src/chat_templates/tool_chat_template_llama3.1_json.jinja";;
  *mistral*)  TP="--tool_call_parser mistral"; CT="--chat-template src/chat_templates/tool_chat_template_mistral.jinja";;
  *)          TP="--tool_call_parser hermes";;
esac

# Context window configuration
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=9216
MAX_CONTEXT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))
VLLM_CONTEXT_LENGTH=$((MAX_CONTEXT_LENGTH + 4096))  # not strictly needed, but so we don't get context window errors

CUDA_VISIBLE_DEVICES=0 uv run trl vllm-serve-async \
    --model "$MODEL_NAME" \
    --max-model-len $VLLM_CONTEXT_LENGTH \
    --disable_log_stats \
    --gpu-memory-utilization 0.94 \
    --enable_auto_tool_choice \
    $CT \
    $RP $TP \
    &


CUDA_VISIBLE_DEVICES=1 uv run accelerate launch \
    --main_process_port $MASTER_PORT \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    --module src.train_grpo -- \
        run=repo_repair \
        model=$MODEL_CONFIG \
        agent.time_limit=60 \
        grpo=multi_turn_gspo \
        grpo.max_prompt_length=$MAX_PROMPT_LENGTH \
        grpo.max_completion_length=$MAX_COMPLETION_LENGTH \
        grpo.num_generations=4 \
        grpo.generation_batch_size=8 \
        grpo.per_device_train_batch_size=4 \
        grpo.gradient_accumulation_steps=4 \
        grpo.optim="paged_adamw_8bit" \
        "$@"  # pass any additional arguments
