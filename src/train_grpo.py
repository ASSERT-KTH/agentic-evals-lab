import os
import logging
from functools import partial
from dataclasses import dataclass, field
from typing import Optional

import wandb
import hydra
import torch
from litellm import register_model
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from peft import LoraConfig as PEFTLoraConfig
from trl import GRPOConfig as HFGRPOConfig, GRPOTrainer as HFGRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import whoami
from datasets import concatenate_datasets

from src.agents.nano_agent import nano_rollout_func, NanoConfig as AgentConfig
from src.agents.mini_agent import mini_rollout_func
from src.rewards import (
    # reasoning rewards
    partial_reasoning_format_reward_func,
    strict_reasoning_format_reward_func,
    # detection rewards
    categorical_correctness_reward_func,
    # mono repair rewards
    sr_diff_format_reward_func,
    sr_diff_similarity_reward_func,
    # repo repair rewards
    unified_diff_similarity_reward_func,
)
from src.data import get_stack_repair_dataset, get_primevul_repair_dataset, get_primevul_detection_dataset, get_swe_gym_repo_repair_dataset
from src.utils.git import resolve_git_commit_hash

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # This ensures output goes to stdout/stderr
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

for noisy in ("httpx", "LiteLLM"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

@dataclass
class RunConfig:
    wandb_project: str = "TTC"
    task_type: str = "repo_repair"
    dataset_type: str = "stack"
    dataset_name: Optional[str] = None
    context_lines: int = 0  # number of context lines to include in diffs
    commit_hash: str = ""  # added at runtime
    push_to_hub: bool = True

    def __post_init__(self):
        if self.task_type not in ["detection", "repair", "repo_repair"]:
            raise ValueError("task_type must be either 'detection' or 'repair'")
        if self.dataset_type not in ["primevul", "stack", "swe_gym"]:
            raise ValueError("dataset_type must be either 'stack', 'primevul' or 'swe_gym'")

@dataclass
class ModelConfig:
    # Transformers configuration
    model_name: str = "Qwen/Qwen3-8B"
    attn_implementation: str = "flash_attention_2"
    chat_template: Optional[str] = None  # optional path to jinja chat template
    # LoRA configuration
    lora: bool = True
    r: int = 32
    lora_alpha: int = 64
    target_modules: tuple[str] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

@dataclass
class GRPOConfig:
    # vLLM generation settings
    use_vllm: bool = True
    vllm_mode: str = "async_server"
    vllm_server_host: str = "127.0.0.1"

    # whether completions are multi-turn or single-turn
    multi_turn: bool = True
    # whether to mask tool responses in the loss
    mask_tool_responses: bool = False

    # Optimizer settings
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "constant_with_warmup"  # or linear, cosine learning rates have been shown to be bad for GRPO, see discussion: https://x.com/kalomaze/status/1895549497692090573
    optim: str = "paged_adamw_8bit"
    
    # Model settings - these will be automatically determined based on GPU architecture
    # when using the custom resolvers in the YAML config
    bf16: bool = True
    fp16: bool = False
    disable_dropout: bool = True

    # Generation and Training settings
    num_generations: int = 4
    generation_batch_size: int = 4
    num_iterations: int = 1  # inner loop \mu in the algorithm, turned off unless >1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_prompt_length: int = 256
    max_completion_length: int = 256

    # Clipping parameters
    epsilon: float = 0.2
    epsilon_high: Optional[float] = None

    # GRPO settings
    beta: float = 0.0  # i.e. no reference_model

    # Reward settings
    scale_rewards: bool = False  # from Dr. GRPO, reward scaling introduces question-level difficulty bias
    
    # Loss type
    loss_type: str = "dr_grpo"  # been shown to have less sequence-length bias
    
    # Attention kernel
    use_liger_loss: bool = True  # should cut memory footprint

    # Gradient checkpointing
    gradient_checkpointing: bool = True  # offload gradient to CPU for better memory utilization
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": True})

    # Training loop settings
    num_train_epochs: int = 3
    max_steps: int = -1
    save_steps: int = 50
    logging_steps: int = 1
    save_total_limit: int = 5
    max_grad_norm: float = 0.1
    resume_from_checkpoint: Optional[str] = None

    # Logging settings
    report_to: str = "wandb"
    run_name: str = ""  # required at runtime
    log_completions: bool = True

    # silence peft warnings
    label_names: list[str] = field(default_factory=lambda: ["labels"])

    ddp_find_unused_parameters: bool = False  # Safe when working on dense LLMs, MoE would be problematic
    ddp_bucket_cap_mb: int = 16

@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

# Register the config schema
cs = ConfigStore.instance()
cs.store(name="base_grpo_config", node=Config, group="")
OmegaConf.register_new_resolver("resolve_git_commit_hash", resolve_git_commit_hash)


@hydra.main(version_base="1.1", config_path="conf", config_name="grpo_config")
def main(cfg: Config) -> None:
    try:
        whoami()
    except Exception:
        raise ValueError("Not logged in to HuggingFace. Please run 'huggingface-cli login' first.")
    
    # Validate that run_name is provided and not empty
    if not cfg.grpo.run_name or cfg.grpo.run_name.strip() == "":
        raise ValueError(
            "run_name is required and cannot be empty. "
            "Please provide a unique run name to prevent model overwriting. "
            "Example: grpo.run_name='my-grpo-experiment-v1'"
        )
        
    # Print all configs for debugging/verification
    logger.info("=" * 50)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Run Config:\n{OmegaConf.to_yaml(cfg.run)}")
    logger.info(f"Model Config:\n{OmegaConf.to_yaml(cfg.model)}")
    logger.info(f"GRPO Config:\n{OmegaConf.to_yaml(cfg.grpo)}")
    logger.info(f"Agent Config:\n{OmegaConf.to_yaml(cfg.agent)}")
    logger.info("=" * 50)
    
    os.environ["WANDB_PROJECT"] = cfg.run.wandb_project

    # Log precision settings
    precision_mode = torch.bfloat16 if cfg.grpo.bf16 else torch.float16 if cfg.grpo.fp16 else torch.float32
    logger.info(f"Training with {precision_mode} precision based on GPU architecture")

    # Load base model
    logger.info(f"Loading model: {cfg.model.model_name}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name, attn_implementation=cfg.model.attn_implementation, torch_dtype=precision_mode)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    if cfg.model.chat_template: tokenizer.chat_template = open(cfg.model.chat_template).read()
        
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # by padding a batch of prompts on the left side we can generate many completions in parallel (padding tokens are masked away)

    if cfg.model.lora:
        lora_params = OmegaConf.to_container(cfg.model, resolve=True)
        lora_config = PEFTLoraConfig(
            r=lora_params["r"],
            lora_alpha=lora_params["lora_alpha"],
            target_modules=lora_params["target_modules"],
            task_type="CAUSAL_LM"
        )
    else:
        lora_config = None

    rollout_func = None
    # Get dataset based on the task
    if cfg.run.task_type == "repair":
        get_repair_dataset = get_stack_repair_dataset if cfg.run.dataset_type == "stack" else get_primevul_repair_dataset
        dataset = get_repair_dataset(
            tokenizer=tokenizer,
            max_prompt_length=cfg.grpo.max_prompt_length,
            context_lines=cfg.run.context_lines
        )
        reward_functions = [
            partial_reasoning_format_reward_func,
            strict_reasoning_format_reward_func,
            sr_diff_format_reward_func,
            sr_diff_similarity_reward_func, 
        ]
        reward_weights = [0.1, 0.2, 0.3, 0.4]
    elif cfg.run.task_type == "detection":  # primevul only
        if not cfg.run.dataset_type == "primevul": raise ValueError("Only primevul supports detection task")
        dataset = get_primevul_detection_dataset(
            tokenizer=tokenizer, 
            max_prompt_length=cfg.grpo.max_prompt_length
        )
        reward_functions = [
            partial_reasoning_format_reward_func,
            strict_reasoning_format_reward_func,
            categorical_correctness_reward_func,
        ]
        reward_weights = [0.1, 0.2, 0.7]
    elif cfg.run.task_type.startswith("repo_repair"):
        if cfg.run.task_type == "repo_repair_multilingual":  # a bit hacky
            dataset_a = get_swe_gym_repo_repair_dataset(dataset_name="SWE-Gym/SWE-Gym").select(range(750))  # pick 750
            dataset_b = get_swe_gym_repo_repair_dataset(dataset_name="SWE-bench/SWE-bench_Multilingual").select(range(250))  # use 250, leave 50 for evals
            dataset = concatenate_datasets([dataset_a, dataset_b])
            dataset = dataset.shuffle(seed=42)
            # Filter out problematic fastlane entry that causes cloning failures with large files
            original_size = len(dataset)
            dataset = dataset.filter(lambda x: not (x.get("repo") == "fastlane/fastlane"))
            logger.info(f"Filtered out problematic entries: {original_size} -> {len(dataset)}")
        else:
            dataset = get_swe_gym_repo_repair_dataset(dataset_name=cfg.run.dataset_name)
        
        # Update agent config with model and token_limit
        cfg.agent.model = f"hosted_vllm/{cfg.model.model_name}"
        cfg.agent.token_limit = cfg.grpo.max_prompt_length + cfg.grpo.max_completion_length
        # Convert OmegaConf to NanoConfig dataclass
        agent_config = AgentConfig(**OmegaConf.to_container(cfg.agent, resolve=True))
        if agent_config.agent_kind == "nano":
            # Extract dataset_name from config for nano_rollout_func
            # FIXME: the multilingual mixed config/dataset is not supported yet with the apptainer backend and will throw an error if used
            dataset_name = cfg.run.dataset_name
            rollout_func = partial(nano_rollout_func, config=agent_config, dataset_name=dataset_name)
        elif agent_config.agent_kind == "mini":
            register_model({
                f"{cfg.model.model_name}": {
                    "mode": "chat",
                    "litellm_provider": "openai",
                    "max_tokens": cfg.grpo.max_prompt_length + cfg.grpo.max_completion_length + 2048,          # set to your served context
                    "input_cost_per_token": 0.0,  # pick any numbers for local
                    "output_cost_per_token": 0.0
                }
            })
            rollout_func = partial(mini_rollout_func, config=agent_config)
        else:
            raise ValueError(f"Unsupported repo repair agent '{agent_config.agent_kind}'")
        # Use a single primary reward (diff similarity) plus a tiny continuous terminal shaping term
        reward_functions = [
            unified_diff_similarity_reward_func,    # primary objective
            # terminal_debugging_habits_reward_func,  # small, continuous shaping to avoid collapse
        ]
        reward_weights = [1]# , 0.1]
    else:
        raise ValueError(f"Unknown task: {cfg.run.task_type}")  # can't happen but looks nice

    # Convert grpo config from OmegaConf to regular Python dict to ensure JSON serialization works
    grpo_params = OmegaConf.to_container(cfg.grpo, resolve=True)
    grpo_params["reward_weights"] = reward_weights
    grpo_params["output_dir"] = f"outputs/{cfg.grpo.run_name}"
    
    # Extract resume_from_checkpoint before creating training args (it's not a TrainingArguments parameter)
    resume_checkpoint = grpo_params.pop("resume_from_checkpoint", None)
    training_args = HFGRPOConfig(**grpo_params)

    logger.info(f"Resuming from checkpoint: {resume_checkpoint}")

    # Initialize trainer with task-specific reward functions
    trainer = HFGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        rollout_func=rollout_func,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config
    )

    # Properly pass the checkpoint path if provided; otherwise start fresh
    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    # Save with task-specific name
    model_save_path = f"grpo_{cfg.run.task_type}_model"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"LoRA adapters saved to {model_save_path}")
    
    # If using LoRA, also save the merged model for simplified VLLM deployment
    if cfg.model.lora:
        merged_model_dir = f"{model_save_path}_merged"
        logger.info(f"Merging LoRA adapters and saving merged model to {merged_model_dir}")
        
        # Get the wrapped trainer model and merge adapters
        peft_model = trainer.model
        merged_model = peft_model.merge_and_unload()

        # Save the merged model
        merged_model.save_pretrained(merged_model_dir)
        tokenizer.save_pretrained(merged_model_dir)
        logger.info(f"Successfully saved merged model to {merged_model_dir}")
    
    # Push to hub if requested
    if cfg.run.push_to_hub:
        model_name = f"ASSERT-KTH/{cfg.grpo.run_name}"
        if cfg.model.lora:
            logger.info(f"Pushing merged model to HuggingFace Hub: {model_name}")
            merged_model.push_to_hub(model_name, tokenizer=tokenizer, commit_message="GRPO training completed")
            logger.info("Successfully pushed merged model to HuggingFace Hub")
        else:
            logger.info(f"Pushing model to HuggingFace Hub: {model_name}")
            trainer.push_to_hub(commit_message="GRPO training completed")
            logger.info("Successfully pushed model to HuggingFace Hub")

if __name__ == "__main__":
    main() 