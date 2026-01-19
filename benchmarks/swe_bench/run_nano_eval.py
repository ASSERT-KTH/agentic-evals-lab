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

import yaml

# Add parent dir to path to import nano_agent
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.nano_agent import _process_one, NanoConfig
from datasets import load_dataset


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_evaluation(config_dict: dict, endpoint: Optional[str] = None, model_name: Optional[str] = None, 
                   subset: Optional[str] = None, split: Optional[str] = None, slice_spec: Optional[str] = None, 
                   output_dir: Optional[Path] = None, backend: Optional[str] = None):
    """Run nano_agent on SWE-bench tasks and save predictions using a process pool."""
    
    # Extract config values, allowing CLI overrides
    agent_cfg = config_dict.get('agent', {})
    eval_cfg = config_dict.get('eval', {})
    model_cfg = config_dict.get('model', {})
    endpoint_cfg = config_dict.get('endpoint', {})
    
    # Use CLI args if provided, otherwise use config
    subset = subset or eval_cfg.get('subset', 'verified')
    split = split or eval_cfg.get('split', 'test')
    slice_spec = slice_spec or eval_cfg.get('slice')
    backend = backend or agent_cfg.get('backend', 'local')
    
    # Derive model name if not provided
    if model_name is None:
        model_name_from_config = model_cfg.get('model_name')
        if model_name_from_config is None:
            # Auto-derive from base_model
            base_model = model_cfg.get('base_model', '')
            model_name_from_config = base_model
        model_name = endpoint_cfg.get('model_name_format', 'hosted_vllm/{MODEL_NAME}').format(
            MODEL_NAME=model_name_from_config
        )
    
    # Derive endpoint if not provided
    if endpoint is None:
        port = config_dict.get('job', {}).get('port', 8000)
        endpoint = endpoint_cfg.get('base_url', 'http://localhost:{PORT}/v1').format(PORT=port)
    
    # Derive output directory if not provided
    if output_dir is None:
        output_base = eval_cfg.get('output_base_dir', 'swe_bench/')
        scaffold = model_cfg.get('scaffold', 'nano-agent')
        base_model = model_cfg.get('base_model', 'unknown')
        # Sanitize model name for filesystem
        model_tag = base_model.replace('/', '__').replace(' ', '_')
        output_dir = Path(output_base) / f"{scaffold}-{model_tag}"
    else:
        output_dir = Path(output_dir)

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

    # Setup config for nano_agent from YAML config
    config = NanoConfig(
        api_base=endpoint,
        model=model_name,
        token_limit=agent_cfg.get('token_limit', 65536),
        time_limit=agent_cfg.get('time_limit', 600),
        tool_limit=agent_cfg.get('tool_limit', 500),
        temperature=agent_cfg.get('temperature', 1.0),
        top_p=agent_cfg.get('top_p'),
        top_k=agent_cfg.get('top_k'),
        min_p=agent_cfg.get('min_p'),
        thinking=agent_cfg.get('thinking'),
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
    
    # Ensure output directory exists so we can read/write resume files
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = output_dir / "preds.jsonl"
    detailed_file = output_dir / "detailed_predictions.jsonl"
    
    # Load existing predictions (for resume) if present
    predictions: dict[str, dict] = {}
    detailed_predictions: dict[str, dict] = {}
    
    if jsonl_file.exists():
        print(f"Resuming from existing predictions at {jsonl_file}")
        with open(jsonl_file, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                iid = rec.get("instance_id")
                if not iid:
                    continue
                predictions[iid] = {
                    "model_patch": rec.get("model_patch", "") or "",
                    "model_name_or_path": rec.get("model_name_or_path", f"nano-agent-{config.model}"),
                }
    
    if detailed_file.exists():
        print(f"Resuming from existing detailed predictions at {detailed_file}")
        with open(detailed_file, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                iid = rec.get("instance_id")
                det = rec.get("detailed_predictions")
                if not iid or det is None:
                    continue
                detailed_predictions[iid] = det
    
    # Decide which instances still need to be processed:
    # - If there is no existing prediction, or
    # - If existing prediction has empty patch, or
    # - If detailed prediction contains an "error" field
    inputs_to_run: list[dict] = []
    for datum in inputs:
        iid = datum["instance_id"]
        existing_pred = predictions.get(iid)
        existing_detail = detailed_predictions.get(iid)
        has_non_empty_patch = bool(existing_pred and existing_pred.get("model_patch"))
        has_error = bool(existing_detail and isinstance(existing_detail, dict) and existing_detail.get("error"))
        if (not existing_pred) or (not has_non_empty_patch) or has_error:
            inputs_to_run.append(datum)
    
    print(f"{len(inputs) - len(inputs_to_run)} instances already completed, {len(inputs_to_run)} remaining to run.")
    
    # Run remaining instances with a process pool
    max_workers_config = eval_cfg.get('max_workers', 48)
    max_workers = min(max_workers_config, len(inputs_to_run)) if inputs_to_run else 0
    if max_workers == 0:
        print("No new instances to process; writing out consolidated files.")
    else:
        print(f"Starting processing {len(inputs_to_run)} remaining instances with {max_workers} workers...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_instance_id = {
                executor.submit(_process_one, datum, config, dataset_name): datum["instance_id"] for datum in inputs_to_run
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
                if completed % 5 == 0 or completed == len(inputs_to_run):
                    print(f"Progress: {completed}/{len(inputs_to_run)} completed")
    
    # Save predictions in JSONL format for SWE-bench harness
    with open(jsonl_file, "w") as f:
        for datum in inputs:
            instance_id = datum["instance_id"]
            pred_data = predictions.get(instance_id, {
                "model_patch": "",
                "model_name_or_path": f"nano-agent-{config.model}",
            })
            obj = {
                "instance_id": instance_id,
                "model_name_or_path": pred_data["model_name_or_path"],
                "model_patch": pred_data["model_patch"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    print(f"Saved JSONL format to {jsonl_file}")
    
    # Save detailed predictions (entire result dictionaries)
    with open(detailed_file, "w") as f:
        for datum in inputs:
            instance_id = datum["instance_id"]
            det = detailed_predictions.get(instance_id, {})
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
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to YAML config file (e.g., benchmarks/configs/nano_qwen3-32b.yaml)")
    parser.add_argument("--endpoint", type=str, default=None,
                        help="Override API endpoint from config")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Override model name from config")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory from config")
    parser.add_argument("--slice", type=str, default=None,
                        help="Slice specification (e.g., ':10' for first 10, '0:100' for range)")
    
    args = parser.parse_args()
    
    # Load config file
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config_dict = load_config(args.config)
    
    run_evaluation(
        config_dict=config_dict,
        endpoint=args.endpoint,
        model_name=args.model_name,
        output_dir=args.output_dir,
        slice_spec=args.slice,
    )


if __name__ == "__main__":
    main()
