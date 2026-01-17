#!/bin/bash
# Run nano evaluation N times and save to different run directories
# Usage: ./run_nano_eval_multiple.sh --config benchmarks/configs/nano_devstral.yaml --runs 5

set -euo pipefail

# Defaults
CONFIG_FILE=""
NUM_RUNS=1
SLICE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="${2:?}"; shift 2;;
    --runs)
      NUM_RUNS="${2:?}"; shift 2;;
    --slice)
      SLICE="${2:?}"; shift 2;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$CONFIG_FILE" ]]; then
  echo "ERROR: --config is required" >&2
  echo "Usage: $0 --config <config_file> [--runs N] [--slice <slice_spec>]" >&2
  echo "  --config: Path to YAML config file (required)" >&2
  echo "  --runs: Number of runs to execute (default: 1)" >&2
  echo "  --slice: Slice specification, e.g., ':10' for first 10 instances (optional)" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: Config file not found: $CONFIG_FILE" >&2
  exit 1
fi

# Parse config file to get output base directory
parse_config() {
  python3 <<EOF
import yaml
import sys

with open("$CONFIG_FILE", 'r') as f:
    config = yaml.safe_load(f)

# Extract values
model_cfg = config.get('model', {})
eval_cfg = config.get('eval', {})
endpoint_cfg = config.get('endpoint', {})
job_cfg = config.get('job', {})

# Export as shell variables
print(f"BASE_MODEL={model_cfg.get('base_model', '')}")
print(f"SCAFFOLD={model_cfg.get('scaffold', 'nano-agent')}")
print(f"OUTPUT_BASE_DIR={eval_cfg.get('output_base_dir', 'swe_bench/')}")
print(f"MODEL_NAME_FORMAT={endpoint_cfg.get('model_name_format', 'hosted_vllm/{MODEL_NAME}')}")
print(f"PORT={job_cfg.get('port', 8000)}")
print(f"START_SERVER={1 if job_cfg.get('start_server', True) else 0}")
EOF
}

# Load config values
eval "$(parse_config)"

# Build a descriptive run tag: <scaffold>-<model_tag>
sanitize_tag() {
  local s="$1"
  s="${s//\//__}"
  s="${s// /_}"
  s=$(printf '%s' "$s" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_+|_+$//g')
  printf '%s' "$s"
}

MODEL_TAG=$(sanitize_tag "$BASE_MODEL")
RUN_TAG="${SCAFFOLD}-${MODEL_TAG}"
BASE_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_TAG}"

echo "=========================================="
echo "Running $NUM_RUNS evaluation run(s)"
echo "Config: $CONFIG_FILE"
echo "Output base: $BASE_OUTPUT_DIR"
echo "=========================================="

# Run evaluation N times
for ((run=0; run<NUM_RUNS; run++)); do
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/run_${run}"
  
  echo ""
  echo "----------------------------------------"
  echo "Starting run $((run + 1))/$NUM_RUNS"
  echo "Output directory: $OUTPUT_DIR"
  echo "----------------------------------------"
  
  # Build eval command
  EVAL_CMD=(uv run python benchmarks/swe_bench/run_nano_eval.py \
    --config "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR")
  
  if [[ -n "$SLICE" ]]; then
    EVAL_CMD+=(--slice "$SLICE")
  fi
  
  # Run the evaluation
  if "${EVAL_CMD[@]}"; then
    echo "✓ Run $((run + 1))/$NUM_RUNS completed successfully"
    echo "  Predictions saved to $OUTPUT_DIR/preds.jsonl"
  else
    echo "✗ Run $((run + 1))/$NUM_RUNS failed"
    exit 1
  fi
done

echo ""
echo "=========================================="
echo "All $NUM_RUNS runs completed successfully!"
echo "Results saved to: $BASE_OUTPUT_DIR/run_*/"
echo "=========================================="
