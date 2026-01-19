#!/usr/bin/env bash
set -euo pipefail

# Run SWE-bench harness evaluation for all r2egym JSON prediction files.
# Usage:
#   benchmarks/swe_bench/run_all_r2egym_evals.sh [--max-workers 8] [--dry-run]
#
# This script finds all *r2egym*.json files in the project root and runs the
# harness evaluation for each one sequentially.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/eval_logs"

max_workers="16"
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-workers)
      max_workers="$2"; shift 2;;
    --dry-run)
      dry_run=true; shift;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

# Create logs directory
mkdir -p "$LOGS_DIR"

# Find all r2egym JSON files in the project root
mapfile -t pred_files < <(find "$PROJECT_ROOT" -maxdepth 1 -name "*r2egym*.json" -type f | sort)

if [[ ${#pred_files[@]} -eq 0 ]]; then
  echo "No *r2egym*.json files found in $PROJECT_ROOT"
  exit 1
fi

echo "Found ${#pred_files[@]} r2egym prediction files to evaluate"
echo "Max workers per job: $max_workers"
echo "Logs directory: $LOGS_DIR"
echo ""

for pred_file in "${pred_files[@]}"; do
  # Extract run ID from filename: qwen3_32b__r2egym__run_0.json -> qwen3_32b__r2egym__run_0
  # Add random suffix to avoid caching issues with the eval harness
  filename=$(basename "$pred_file" .json)
  random_suffix=$(head -c 100 /dev/urandom | tr -dc 'a-z0-9' | head -c 8)
  run_id="${filename}__${random_suffix}"
  
  log_file="$LOGS_DIR/${run_id}.out"
  
  echo "Run: $run_id"
  echo "  Predictions: $pred_file"
  echo "  Log file: $log_file"
  
  if [[ "$dry_run" == true ]]; then
    echo "  [DRY RUN] Would execute:"
    echo "    $SCRIPT_DIR/run_harness_eval.sh --preds $pred_file --run-id $run_id --max-workers $max_workers 2>&1 | tee $log_file"
  else
    echo "  Running..."
    "$SCRIPT_DIR/run_harness_eval.sh" \
      --preds "$pred_file" \
      --run-id "$run_id" \
      --max-workers "$max_workers" \
      2>&1 | tee "$log_file"
    echo "  Completed."
  fi
  echo ""
done

echo "All evaluations completed."
