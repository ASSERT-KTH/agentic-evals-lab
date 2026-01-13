#!/bin/bash
#SBATCH --job-name=crrl-swe-nano
#SBATCH --output=logs/swe_nano_%A_%a.out
#SBATCH --error=logs/swe_nano_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus 8
#SBATCH --time=12:00:00
#SBATCH -C "fat"
#SBATCH --array=0

set -euo pipefail

# Defaults
BASE_MODEL="Qwen/Qwen3-32B"     # HF model to serve with vLLM
LORA_PATH=""                    # Optional LoRA path; adapter name auto-derived from basename if set
MODEL_NAME=""                   # Model name passed to the agent; auto-derived if empty
SCAFFOLD="nano-agent"           # Scaffold identifier for run tagging
OUTPUT_BASE_DIR="swe_bench/"
SUBSET="verified"
SPLIT="test"
SLICE=""
PORT=8000
START_SERVER=1
WANDB_API_KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model)
      BASE_MODEL="${2:?}"; shift 2;;
    --lora-path)
      LORA_PATH="${2:?}"; shift 2;;
    --model-name)
      MODEL_NAME="${2:?}"; shift 2;;
    --output-dir)
      OUTPUT_BASE_DIR="${2:?}"; shift 2;;
    --subset)
      SUBSET="${2:?}"; shift 2;;
    --split)
      SPLIT="${2:?}"; shift 2;;
    --slice)
      SLICE="${2:?}"; shift 2;;
    --scaffold)
      SCAFFOLD="${2:?}"; shift 2;;
    --port)
      PORT="${2:?}"; shift 2;;
    --no-server)
      START_SERVER=0; shift;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

# Array task ID is used as the run index (for variance measurement with multiple runs)
# e.g., --array=0-9 gives 10 independent runs stored in run_0/ through run_9/
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Stagger job starts to avoid HuggingFace cache lock contention when many jobs
# try to load the same model simultaneously (30s delay per task)
if [[ $TASK_ID -gt 0 ]]; then
  STAGGER_DELAY=$(( TASK_ID * 30 ))
  echo "Staggering start by ${STAGGER_DELAY}s (task $TASK_ID) to avoid cache contention..."
  sleep "$STAGGER_DELAY"
fi

# Offset port to avoid conflicts if multiple tasks land on the same node
if [[ $START_SERVER -eq 1 ]]; then
  PORT=$(( PORT + TASK_ID ))
fi

ENDPOINT="http://localhost:${PORT}/v1"

# Derive MODEL_NAME and LoRA adapter name if not explicitly provided
LORA_ADAPTER_NAME=""
if [[ -n "$LORA_PATH" ]]; then
  if [[ -z "$MODEL_NAME" ]]; then
    # Derive adapter name from LoRA path basename and sanitize for use as model id
    ADAPTER_BASENAME="$(basename "$LORA_PATH")"
    LORA_ADAPTER_NAME=$(printf '%s' "$ADAPTER_BASENAME" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_+|_+$//g')
    MODEL_NAME="$LORA_ADAPTER_NAME"
  else
    # Honor explicit model name; use it as the LoRA adapter name
    LORA_ADAPTER_NAME="$MODEL_NAME"
  fi
else
  # No LoRA: default model name is the base model being served
  if [[ -z "$MODEL_NAME" ]]; then
    MODEL_NAME="$BASE_MODEL"
  fi
fi

# Build a descriptive run tag: <scaffold>-<model_tag>
sanitize_tag() {
  local s="$1"
  s="${s//\//__}"
  s="${s// /_}"
  s=$(printf '%s' "$s" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_+|_+$//g')
  printf '%s' "$s"
}

if [[ -n "$LORA_PATH" ]]; then
  BASE_TAG=$(sanitize_tag "$BASE_MODEL")
  ADAPTER_TAG=$(sanitize_tag "$(basename "$LORA_PATH")")
  MODEL_TAG="${BASE_TAG}__lora__${ADAPTER_TAG}"
else
  MODEL_TAG=$(sanitize_tag "$MODEL_NAME")
fi

RUN_TAG="${SCAFFOLD}-${MODEL_TAG}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_TAG}/run_${TASK_ID}"

mkdir -p "$(dirname "logs/.keep")" "$OUTPUT_DIR"

wait_for_vllm() {
  local url="$1"; local -i tries=180
  while (( tries-- > 0 )); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "$url/models" || true)
    if [[ "$code" == "200" ]]; then return 0; fi
    sleep 10
  done
  return 1
}

# Minimal parser selection based on base model and optional chat template
RP=""; TP=""; CT=""
case "${BASE_MODEL,,}" in
  *qwen*)     RP="--reasoning-parser deepseek_r1"; TP="--tool-call-parser hermes";;
  *nemotron*) TP="--tool-call-parser llama3_json"; CT="--chat-template src/chat_templates/tool_chat_template_llama3.1_json.jinja";;
  *llama*)    TP="--tool-call-parser llama3_json"; CT="--chat-template src/chat_templates/tool_chat_template_llama3.1_json.jinja";;
  *mistral*)  TP="--tool-call-parser mistral"; CT="--chat-template src/chat_templates/tool_chat_template_mistral.jinja";;
  *)          TP="--tool-call-parser hermes";;
esac

VLLM_PID=""
export MAX_CONTEXT_LEN=65536
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
if [[ $START_SERVER -eq 1 ]]; then
  echo "Starting vLLM server on port $PORT for base model '$BASE_MODEL'..."
  CMD=(uv run vllm serve $BASE_MODEL \
    --port "$PORT" \
    --enable-auto-tool-choice \
    --tensor-parallel-size 8 \
    --max-model-len $MAX_CONTEXT_LEN \
    --hf-overrides '{"max_position_embeddings": '$MAX_CONTEXT_LEN'}' \
    --enable_prefix_caching \
    $CT \
    $RP $TP)

  if [[ -n "$LORA_PATH" ]]; then
    CMD+=(--max-lora-rank 32 --enable-lora --lora-modules "$LORA_ADAPTER_NAME=$LORA_PATH")
  fi

  # Start server in background and capture PID
  (exec "${CMD[@]}") > "logs/vllm_${SLURM_JOB_ID:-$$}.log" 2>&1 &
  VLLM_PID=$!
  trap 'if [[ -n "$VLLM_PID" ]]; then kill "$VLLM_PID" 2>/dev/null || true; fi' EXIT

  echo "Waiting for vLLM to become ready at $ENDPOINT ..."
  if ! wait_for_vllm "$ENDPOINT"; then
    echo "vLLM did not become ready in time" >&2
    exit 1
  fi
fi

echo "Running nano_agent evaluation (run $TASK_ID) with model '$MODEL_NAME'..."

# Build eval command, only include --slice if explicitly provided
EVAL_CMD=(uv run python benchmarks/swe_bench/run_nano_eval.py \
  --endpoint "$ENDPOINT" \
  --model-name "hosted_vllm/$MODEL_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --subset "$SUBSET" \
  --split "$SPLIT" \
  --backend "apptainer")

if [[ -n "$SLICE" ]]; then
  EVAL_CMD+=(--slice "$SLICE")
fi

OPENAI_API_BASE="$ENDPOINT" \
OPENAI_API_KEY="dummy" \
"${EVAL_CMD[@]}"

echo "Predictions saved to $OUTPUT_DIR/preds.jsonl"

# Stop vLLM if we started it
if [[ -n "$VLLM_PID" ]]; then
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
fi


