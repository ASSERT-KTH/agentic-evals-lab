#!/bin/bash
#SBATCH --job-name=pull-swegym-images
#SBATCH --output=logs/pull_swegym_%A_%a.out
#SBATCH --error=logs/pull_swegym_%A_%a.err
#SBATCH --nodes=1
#SBATCH -C "thin"
#SBATCH --gpus 1
#SBATCH --time=06:00:00
#SBATCH --array=0-19

set -euo pipefail

# Configuration
DATASET="SWE-Gym/SWE-Gym"
SPLIT="train"
TOTAL_WORKERS=20
WORKER_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "Starting worker ${WORKER_ID}/${TOTAL_WORKERS} for SWE-Gym ${DATASET}..."

# Use a unique temp file for this job to avoid conflicts
TEMP_SIF="/proj/berzelius-2024-336/users/x_andaf/CodeRepairRL/temp_pull_swegym_${SLURM_JOB_ID}_${WORKER_ID}.sif"

uv run scripts/pull_swegym_images.py \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --temp-sif "$TEMP_SIF" \
  --shard-id "$WORKER_ID" \
  --num-shards "$TOTAL_WORKERS"

echo "Worker ${WORKER_ID} finished."

