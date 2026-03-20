# agentic-evals-lab

A framework for training and running LLMs with reinforcement learning in agentic settings. Covers the full pipeline: GRPO/SFT training, agent inference, and agentic evaluation.

**Artifacts** (trajectories, eval results): [ASSERT-KTH/agentic-evals-artifacts](https://huggingface.co/datasets/ASSERT-KTH/agentic-evals-artifacts) on Hugging Face.

## Citation

If you use this framework, please cite our paper:

```bibtex
@article{bjarnason2026randomness,
  title={On Randomness in Agentic Evals},
  author={Bjarnason, Bjarni Haukur and Silva, Andr{\'e} and Monperrus, Martin},
  journal={arXiv preprint arXiv:2602.07150},
  year={2026}
}
```

Paper: [arXiv:2602.07150](https://arxiv.org/abs/2602.07150)

## Getting Started

### Building the Container

To build the Apptainer container:

```bash
# Build the training container 
apptainer build crrl.sif scripts/train_container.def
```

(the build process may take several minutes)

## Reproducing on different compute setups

### Using our Apptainer/SLURM setup

Before launching jobs, you should set `CRRL_WORKDIR` in your environment. Otherwise large files like model weights are downloaded to your `$HOME/.cache`:

```bash
# Choose your working directory (pick a location with plenty of fast storage)
export CRRL_WORKDIR="/path/to/your/crrl_workspace"
export WANDB_API_KEY="your-key"
```

Then follow the container build and SLURM job submission steps above. This ensures that large model files and datasets are stored in a location with sufficient space rather than your home directory.

### Alternative: Local reproduction with uv

If you do not have Apptainer/SLURM or want to reproduce runs locally, you can use `uv`. Below are self-contained bash snippets.

### 1) Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2) Create the environment and install dependencies

```bash
# Install project dependencies (creates/uses a virtualenv automatically)
uv sync --extra vllm --extra flash
```

### 3.) Exact 14B GRPO reproduction (3x ≥80GB GPUs) — run in two terminals

- Requires 3 GPUs with at least 80 GB VRAM each (e.g., A100 80GB/H100 80GB)
- Terminal 1 runs the vLLM server on GPU 0; Terminal 2 runs training on GPUs 1–2

Terminal 1 (vLLM server on GPU 0):

```bash
CUDA_VISIBLE_DEVICES=0 uv run trl vllm-serve-async \
  --model "Qwen/Qwen3-14B" \
  --max-model-len 14336 \
  --gpu-memory-utilization 0.94 \
  --async-scheduling \
  --enable-prefix-caching \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 2048 \
  --disable_log_stats \
  --enable_auto_tool_choice \
  --reasoning_parser qwen3 \
  --tool_call_parser hermes
# Leave this terminal running
```

Terminal 2 (trainer on GPUs 1–2):

```bash
CUDA_VISIBLE_DEVICES=1,2 uv run accelerate launch \
  --config_file scripts/deepspeed/zero2.yaml \
  --num_processes 2 \
  --module src.train_grpo -- \
        run=repo_repair \
        model=medium_qwen \
        agent.time_limit=60 \
        grpo=multi_turn_gspo \
        grpo.max_prompt_length=1024 \
        grpo.max_completion_length=12288 \
        grpo.num_train_epochs=10 \
        grpo.num_generations=8 \
        grpo.generation_batch_size=8 \
        grpo.per_device_train_batch_size=4 \
        grpo.gradient_accumulation_steps=4 \
        grpo.optim=adamw_torch \
        grpo.run_name="your-run-name"
```

Notes:
- If you plan to push to the HuggingFace Hub, run `huggingface-cli login` first and drop `run.push_to_hub=false`.
- You can override any config at the CLI via Hydra (e.g., change model, learning rate, batch sizes, etc.).

### Running Supervised Fine-Tuning (SFT)

Before GRPO training, you can optionally run SFT to create a better starting point:

```bash
# Run SFT training job (small model)
sbatch scripts/small_sft_lora_train_job.sh

# Run SFT training job (large model)
sbatch scripts/large_sft_lora_train_job.sh

# Or run locally for testing
uv run -m src.train_sft
```

The SFT stage uses curated datasets of high-quality code repair examples to provide the model with a strong foundation before RL training.

### Running GRPO Training Jobs

We provide specialized SLURM scripts for different model sizes, each pre-configured with appropriate compute resource allocations:

```bash
# For small models (8B), defaults to Qwen/Qwen3-8B
sbatch scripts/grpo/small_grpo_lora_train_job.sh grpo.run_name="custom-experiment-name"  # LoRA training (3 GPUs)

# For medium models (32B), defaults to Qwen/Qwen3-14B
sbatch scripts/grpo/medium_grpo_lora_train_job.sh grpo.run_name="custom-experiment-name"  # LoRA training (3 GPUs)
```

Each script includes pre-tuned GRPO parameters optimized for the corresponding model size category. The scripts support three task types:
- **detection**: Binary vulnerability detection
- **repair**: Single-file code repair with search-replace diffs
- **repo_repair**: Repository-level code repair using agentic approaches

You can customize training with Hydra overrides:

```bash
# Change task type
sbatch scripts/grpo/medium_grpo_lora_train_job.sh run=detection

# Use a different model
sbatch scripts/grpo/medium_grpo_train_job.sh model=medium_llama
```

### Model selection via MODEL_CONFIG (env)

You can select a model configuration by setting the `MODEL_CONFIG` environment variable before submitting the job. The value should match a file in `src/conf/model/` (without the `.yaml` suffix).

Example:

```bash
MODEL_CONFIG=small_qwen \
sbatch scripts/grpo/medium_grpo_lora_train_job.sh \
  grpo.run_name="Qwen3-8B-Multingual"
```

Notes:
- Small and medium scripts respect `MODEL_CONFIG`. Large scripts are fixed to Qwen3 models.
- vLLM parser/templating is auto-selected in scripts based on the base model name (Qwen → qwen3/hermes; Llama → llama3_json + llama3.1 tool chat template etc.)

## Local Development

For "local" development and testing without Apptainer containers, you can use `uv` directly.

### Installing uv

Install the `uv` package manager with:

MacOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (project not tested on Windows)
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Testing

```bash
# run all tests
uv run pytest

# run specific testing file
uv run pytest tests/test_search_replace_diff.py

# run specific test
uv run pytest tests/test_search_replace_diff.py::test_specific_function
```

## Documentation Structure

This repository uses several Markdown files to organize information:

- **README.md**: (This file) Provides a high-level overview, setup instructions, and basic usage examples.
- **docs/PROJECT.md**: Contains detailed information about the project's goals, implementation notes, theoretical background, and conceptual insights.
- **docs/DIARY.md**: A development diary tracking progress, challenges, and decisions.
- **docs/AGENT_RL_INTEGRATION.md**: Describes our approach to integrating agent frameworks into RL training loops using OpenAI-compatible API servers.
- **docs/DATASETS.md**: Describes the datasets used in the project.
- **docs/RESOURCES.md**: Lists relevant research papers, literature and broader resources reviewed for the project.
- **docs/VOCABULARY.md**: Defines key terms and concepts used throughout the project.
- **docs/PAPER.md**: Outlines the structure and key points for the academic paper.
