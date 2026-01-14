#!/usr/bin/env python3
"""
Run SWE-bench evaluation using mini_agent.py (Mini-SWE-Agent backend)

Interface mirrors run_nano_eval.py to keep flows consistent across agents.
"""

import json
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import yaml

# Add project root to path to import mini_agent
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.mini_agent import _process_one, AgentConfig  # type: ignore
from datasets import load_dataset


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_evaluation(config_dict: dict, endpoint: Optional[str] = None, model_name: Optional[str] = None,
                   subset: Optional[str] = None, split: Optional[str] = None, slice_spec: Optional[str] = None,
                   output_dir: Optional[Path] = None):
    """Run mini_agent on SWE-bench tasks and save predictions using a process pool."""
    
    # Extract config values, allowing CLI overrides
    agent_cfg = config_dict.get('agent', {})
    eval_cfg = config_dict.get('eval', {})
    model_cfg = config_dict.get('model', {})
    endpoint_cfg = config_dict.get('endpoint', {})
    
    # Use CLI args if provided, otherwise use config
    subset = subset or eval_cfg.get('subset', 'verified')
    split = split or eval_cfg.get('split', 'test')
    slice_spec = slice_spec or eval_cfg.get('slice', ':25')
    
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
        scaffold = model_cfg.get('scaffold', 'mini-agent')
        base_model = model_cfg.get('base_model', 'unknown')
        # Sanitize model name for filesystem
        model_tag = base_model.replace('/', '__').replace(' ', '_')
        output_dir = Path(output_base) / f"{scaffold}-{model_tag}"
    else:
        output_dir = Path(output_dir)

    # Load SWE-bench dataset
    dataset = load_dataset(f"princeton-nlp/SWE-bench_{subset}", split=split)

    # Parse slice
    # Supported forms:
    #   ":N"        -> first N instances
    #   "start:end" -> instances in [start, end) zero-based half-open interval
    if slice_spec:
        if slice_spec.startswith(":"):
            dataset = dataset.select(range(int(slice_spec[1:])))
        elif ":" in slice_spec:
            start_str, end_str = slice_spec.split(":", 1)
            start_idx = int(start_str)
            end_idx = int(end_str)
            if start_idx < 0 or end_idx < 0:
                raise ValueError("slice must be non-negative indices")
            if end_idx < start_idx:
                raise ValueError("slice end must be >= start")
            dataset = dataset.select(range(start_idx, min(end_idx, len(dataset))))

    # Setup config for mini_agent from YAML config
    config = AgentConfig(
        api_base=endpoint,
        model=model_name,
        token_limit=agent_cfg.get('token_limit', 16384),
        time_limit=agent_cfg.get('time_limit', 40),
        tool_limit=agent_cfg.get('tool_limit', 30),
        temperature=agent_cfg.get('temperature', 0.2),
        top_p=agent_cfg.get('top_p'),
        top_k=agent_cfg.get('top_k'),
        min_p=agent_cfg.get('min_p'),
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

    # Run with a process pool
    max_workers_config = eval_cfg.get('max_workers', 8)
    max_workers = min(max_workers_config, len(inputs)) if inputs else 0
    if max_workers == 0:
        print("No instances to process.")
        return

    print(f"Starting processing {len(inputs)} instances with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_instance_id = {
            executor.submit(_process_one, datum, config): datum["instance_id"] for datum in inputs
        }

        completed = 0
        for future in as_completed(future_to_instance_id):
            instance_id = future_to_instance_id[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"Error processing {instance_id}: {e}")
                result = {}

            # Extract model patch; prefer 'generated_diff' produced by mini_agent
            patch = (
                result.get("generated_diff", "")
                or result.get("patch", "")
                or result.get("diff", "")
                or result.get("model_patch", "")
            )

            predictions[instance_id] = {
                "model_patch": patch or "",
                "model_name_or_path": f"mini-agent-{config.model}",
            }

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
    parser = argparse.ArgumentParser(description="Run SWE-bench eval with mini_agent")
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to YAML config file (e.g., benchmarks/configs/mini_qwen3-8b.yaml)")
    
    args = parser.parse_args()
    
    # Load config file
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config_dict = load_config(args.config)
    
    run_evaluation(
        config_dict=config_dict,
    )


if __name__ == "__main__":
    main()


