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
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="${2:?}"; shift 2;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$CONFIG_FILE" ]]; then
  echo "ERROR: --config is required" >&2
  echo "Usage: $0 --config benchmarks/configs/nano_qwen3-32b.yaml" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

# Parse config file using Python
parse_config() {
  python3 <<EOF
import yaml
import sys
import json

with open("$CONFIG_FILE", 'r') as f:
    config = yaml.safe_load(f)

# Extract values
model_cfg = config.get('model', {})
vllm_cfg = config.get('vllm', {})
eval_cfg = config.get('eval', {})
job_cfg = config.get('job', {})
endpoint_cfg = config.get('endpoint', {})

# Export as shell variables
print(f"BASE_MODEL={model_cfg.get('base_model', '')}")
print(f"LORA_PATH={model_cfg.get('lora_path', '') or ''}")
print(f"MODEL_NAME={model_cfg.get('model_name', '') or ''}")
print(f"SCAFFOLD={model_cfg.get('scaffold', 'nano-agent')}")
print(f"OUTPUT_BASE_DIR={eval_cfg.get('output_base_dir', 'swe_bench/')}")
print(f"SUBSET={eval_cfg.get('subset', 'verified')}")
print(f"SPLIT={eval_cfg.get('split', 'test')}")
print(f"SLICE={eval_cfg.get('slice', '') or ''}")
print(f"PORT={job_cfg.get('port', 8000)}")
print(f"MAX_CONTEXT_LEN={vllm_cfg.get('env', {}).get('MAX_CONTEXT_LEN', 65536)}")
print(f"VLLM_ALLOW_LONG_MAX_MODEL_LEN={vllm_cfg.get('env', {}).get('VLLM_ALLOW_LONG_MAX_MODEL_LEN', 1)}")
print(f"VLLM_COMMAND={json.dumps(vllm_cfg.get('command', ''))}")
print(f"LORA_MAX_RANK={vllm_cfg.get('lora', {}).get('max_rank', 32)}")
print(f"MODEL_NAME_FORMAT={endpoint_cfg.get('model_name_format', 'hosted_vllm/{MODEL_NAME}')}")
print(f"START_SERVER={1 if job_cfg.get('start_server', True) else 0}")
EOF
}

# Load config values
eval "$(parse_config)"

# Array task ID is used as the run index (for variance measurement with multiple runs)
# e.g., --array=0-9 gives 10 independent runs stored in run_0/ through run_9/
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Offset port to avoid conflicts if multiple tasks land on the same node
# START_SERVER is loaded from config
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

# Export environment variables for vLLM
export MAX_CONTEXT_LEN
export VLLM_ALLOW_LONG_MAX_MODEL_LEN

# Build LORA_MODULES string for vLLM command template
LORA_MODULES_STR=""
if [[ -n "$LORA_PATH" ]]; then
  LORA_MODULES_STR="--max-lora-rank $LORA_MAX_RANK --enable-lora --lora-modules \"$LORA_ADAPTER_NAME=$LORA_PATH\""
fi

# Build vLLM command from template
VLLM_CMD=$(echo "$VLLM_COMMAND" | \
  sed "s|{BASE_MODEL}|$BASE_MODEL|g" | \
  sed "s|{PORT}|$PORT|g" | \
  sed "s|{MAX_CONTEXT_LEN}|$MAX_CONTEXT_LEN|g" | \
  sed "s|{LORA_MODULES}|$LORA_MODULES_STR|g")

VLLM_PID=""
if [[ $START_SERVER -eq 1 ]]; then
  echo "Starting vLLM server on port $PORT for base model '$BASE_MODEL'..."
  echo "Command: $VLLM_CMD"
  
  # Execute vLLM command in background
  eval "$VLLM_CMD" > "logs/vllm_${SLURM_JOB_ID:-$$}.log" 2>&1 &
  VLLM_PID=$!
  trap 'if [[ -n "$VLLM_PID" ]]; then kill "$VLLM_PID" 2>/dev/null || true; fi' EXIT

  echo "Waiting for vLLM to become ready at $ENDPOINT ..."
  if ! wait_for_vllm "$ENDPOINT"; then
    echo "vLLM did not become ready in time" >&2
    exit 1
  fi
fi

echo "Running nano_agent evaluation (run $TASK_ID) with model '$MODEL_NAME'..."

# Format model name according to config
FORMATTED_MODEL_NAME=$(echo "$MODEL_NAME_FORMAT" | sed "s|{MODEL_NAME}|$MODEL_NAME|g")

# Build eval command, only include --slice if explicitly provided
EVAL_CMD=(uv run python benchmarks/swe_bench/run_nano_eval.py \
  --config "$CONFIG_FILE" \
  --endpoint "$ENDPOINT" \
  --model-name "$FORMATTED_MODEL_NAME" \
  --output-dir "$OUTPUT_DIR")

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


