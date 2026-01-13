#!/usr/bin/env python3
"""
Run SWE-bench evaluation using nano_agent.py

This file intentionally mirrors the interface of run_aider_eval.py to keep
eval flows consistent across agents.
"""

import json
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

# Add parent dir to path to import nano_agent
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.nano_agent import _process_one, NanoConfig
from datasets import load_dataset


def run_evaluation(endpoint: str, model_name: str, subset: str, split: str, slice_spec: Optional[str], output_dir: Path, backend: str = "local"):
    """Run nano_agent on SWE-bench tasks and save predictions using a process pool."""

    # Load SWE-bench dataset
    dataset_name = f"princeton-nlp/SWE-bench_{subset}"
    dataset = load_dataset(dataset_name, split=split)

    # Parse slice
    # Supported forms:
    #   ":N"        -> first N instances
    #   "start:end" -> instances in [start, end) zero-based half-open interval
    #   "start:count" when suffixed with "+" as in "start+count" is NOT supported to avoid ambiguity
    if slice_spec:
        if slice_spec.startswith(":"):
            # first N
            dataset = dataset.select(range(int(slice_spec[1:])))
        elif ":" in slice_spec:
            # start:end range
            start_str, end_str = slice_spec.split(":", 1)
            start_idx = int(start_str)
            end_idx = int(end_str)
            if start_idx < 0 or end_idx < 0:
                raise ValueError("slice must be non-negative indices")
            if end_idx < start_idx:
                raise ValueError("slice end must be >= start")
            dataset = dataset.select(range(start_idx, min(end_idx, len(dataset))))

    # Setup config for nano_agent
    config = NanoConfig(
        api_base=endpoint,
        model=model_name,  # e.g., "nano" for LoRA
        token_limit=65536,
        time_limit=600,
        tool_limit=500,
        # hyper-parameters as used in the qwen-3 tech report
        # temperature=0.6,
        # top_p=0.95,
        # top_k=20,
        # min_p=0,
        # hyper-parameters as suggest by deepswe blog
        temperature=1.0,
        thinking=True,
        backend=backend,
    )

    # Prepare inputs for workers
    inputs: list[dict] = []
    for instance in dataset:
        inputs.append({
            "instance_id": instance["instance_id"],
            "problem_statement": instance["problem_statement"],
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "version": instance.get("version", ""),
        })

    predictions: dict[str, dict] = {}
    detailed_predictions: dict[str, dict] = {}

    # Run with a process pool of up to 8 workers
    max_workers = min(48, len(inputs)) if inputs else 0
    if max_workers == 0:
        print("No instances to process.")
        return

    print(f"Starting processing {len(inputs)} instances with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_instance_id = {
            executor.submit(_process_one, datum, config, dataset_name): datum["instance_id"] for datum in inputs
        }

        completed = 0
        for future in as_completed(future_to_instance_id):
            instance_id = future_to_instance_id[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"Error processing {instance_id}: {e}")
                result = {}

            # Extract model patch; prefer 'generated_diff' produced by nano_agent
            patch = (
                result.get("generated_diff", "")
                or result.get("patch", "")
                or result.get("diff", "")
                or result.get("model_patch", "")
            )

            predictions[instance_id] = {
                "model_patch": patch or "",
                "model_name_or_path": f"nano-agent-{config.model}",
            }

            # Store the entire result dictionary for detailed analysis
            if result:
                detailed_predictions[instance_id] = result

            completed += 1
            if completed % 5 == 0 or completed == len(inputs):
                print(f"Progress: {completed}/{len(inputs)} completed")
    
    # Save predictions in JSONL format for SWE-bench harness
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = output_dir / "preds.jsonl"
    with open(jsonl_file, "w") as f:
        for instance_id, pred_data in predictions.items():
            obj = {
                "instance_id": instance_id,
                "model_name_or_path": pred_data["model_name_or_path"],
                "model_patch": pred_data["model_patch"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    print(f"Saved JSONL format to {jsonl_file}")

    # Save detailed predictions (entire result dictionaries)
    detailed_file = output_dir / "detailed_predictions.jsonl"
    with open(detailed_file, "w") as f:
        for instance_id, det in detailed_predictions.items():
            obj = {"instance_id": instance_id, "detailed_predictions": det}
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    print(f"Saved detailed predictions to {detailed_file}")
    
    # Quick validation - check if patches can apply
    valid_count = 0
    for instance_id, pred_data in predictions.items():
        if pred_data["model_patch"]:
            valid_count += 1
    
    print(f"\nSummary: {valid_count}/{len(predictions)} instances have non-empty patches")


def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench eval with nano_agent")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1",
                        help="Model endpoint URL")
    parser.add_argument("--model-name", default="nano",
                        help="Model name to use (e.g., 'nano' for LoRA, base model name for baseline)")
    parser.add_argument("--output-dir", default="swe_bench/results_nano",
                        help="Output directory for results")
    parser.add_argument("--subset", default="verified",
                        help="SWE-bench subset (verified, lite, full)")
    parser.add_argument("--split", default="test",
                        help="Dataset split")
    parser.add_argument("--slice", default=None,
                        help="Slice to run. Forms: :N (first N) or start:end (half-open)")
    parser.add_argument("--backend", choices=["local", "apptainer"], default="local",
                        help="Execution backend (local or apptainer)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    run_evaluation(args.endpoint, args.model_name, args.subset, args.split, args.slice, output_dir, args.backend)


if __name__ == "__main__":
    main()
