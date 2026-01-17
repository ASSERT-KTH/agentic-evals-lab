import os
import time
import logging
from typing import Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

from nano import Agent
from nano.env import Environment, ApptainerEnvironment

from src.utils.git import handle_to_url, clone_repo_at_commit, clean_repo_dir

logger = logging.getLogger(__name__)


def setup_env_common(env: Environment):
    """
    Common setup steps that apply to all datasets.
    This includes installing system packages and setting up git for commits.
    """
    # Install ripgrep
    env.run_shell("apt-get update && apt-get install -y ripgrep 2>/dev/null || true")

    # Commit all changes to ensure we have a clean state
    env.run_shell("git config --global user.email 'you@example.com'")
    env.run_shell("git config --global user.name 'Your Name'")
    env.run_shell("git add . && git commit -m 'add changes'")


def setup_env_swebench(env: Environment):
    """
    SWE-bench specific environment setup.
    Includes PATH configuration, conda env setup, and package installation.
    """
    repo_path = "/testbed"
    alt_path = "/root"
    
    # Set the PATH for all subsequent commands
    DOCKER_PATH = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    env.path = DOCKER_PATH
    
    # === setup_env_swebench path ===
    # Make run_tests.sh executable (if present)
    # We will not use this script in nano-agent
    # env.run_shell("chmod +x /run_tests.sh 2>/dev/null || true")
    
    # Create symlink of conda env to /root/.venv
    env.run_shell("ln -sf /opt/miniconda3/envs/testbed /root/.venv")
    
    # Install required packages
    env.run_shell("python -m pip install chardet -q")
    
    # === setup_env (non-swebench R2E-Gym) path ===
    # Create local bin directory if needed
    env.run_shell(f"mkdir -p {alt_path}/.local/bin")
    
    # Symlink python executables
    env.run_shell(f"ln -sf {repo_path}/.venv/bin/python {alt_path}/.local/bin/python")
    env.run_shell(f"ln -sf {repo_path}/.venv/bin/python {alt_path}/.local/bin/python3")
    
    # Symlink all executables from venv bin
    env.run_shell(f"find {repo_path}/.venv/bin -type f -executable -exec ln -sf {{}} {alt_path}/.local/bin/ \\;")
    
    # Clean up pycache files
    env.run_shell("find . -name '*.pyc' -delete 2>/dev/null || true")
    env.run_shell("find . -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true")
    
    # Clean up pycache from r2e_tests (if present)
    env.run_shell("find /r2e_tests -name '*.pyc' -delete 2>/dev/null || true")
    env.run_shell("find /r2e_tests -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true")
    
    # Move r2e_tests to /root (if present)
    # We will not use this script in nano-agent
    # env.run_shell(f"mv /r2e_tests {alt_path}/r2e_tests 2>/dev/null || true")
    
    # Create symlink for r2e_tests in repo
    # We will not use this script in nano-agent
    # env.run_shell(f"ln -sf {alt_path}/r2e_tests {repo_path}/r2e_tests 2>/dev/null || true")
    
    # Call common setup
    setup_env_common(env)


def setup_env_swegym(env: Environment):
    """
    SWE-Gym specific environment setup.
    Only calls the common setup since SWE-Gym images are pre-configured.
    """
    setup_env_common(env)


# Supported HuggingFace dataset names
SUPPORTED_DATASETS = {
    "princeton-nlp/SWE-bench_Verified",
    "SWE-Gym/SWE-Gym",
    "SWE-Gym/SWE-Gym-Lite",
}


def _is_swegym_dataset(dataset_name: str) -> bool:
    """Check if the dataset name is a SWE-Gym dataset."""
    return dataset_name.startswith("SWE-Gym/")


def _get_setup_fn(dataset_name: str):
    """
    Get the appropriate setup function for the given dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
    
    Returns:
        Setup function to use for the environment
    """
    if _is_swegym_dataset(dataset_name):
        return setup_env_swegym
    elif dataset_name.startswith("princeton-nlp/SWE-bench"):
        return setup_env_swebench
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets are: {', '.join(sorted(SUPPORTED_DATASETS))}"
        )


def _construct_image_name(instance_id: str, dataset_name: str) -> str:
    """
    Construct the Docker image name for Apptainer backend based on dataset type.
    
    Args:
        instance_id: The instance identifier from the dataset
        dataset_name: HuggingFace dataset name (e.g., "princeton-nlp/SWE-bench_Verified" or "SWE-Gym/SWE-Gym")
    
    Returns:
        Full Docker image name (e.g., "docker.io/slimshetty/swebench-verified:sweb.eval.x86_64.{instance_id}")
    
    Raises:
        ValueError: If dataset_name is not one of the supported datasets
    """
    if _is_swegym_dataset(dataset_name):
        # SWE-Gym format: xingyaoww/sweb.eval.x86_64.{instance_id_with_underscores}
        # Replace "__" with "_s_" in instance_id
        transformed_id = instance_id.replace("__", "_s_").lower()
        image_name = f"xingyaoww/sweb.eval.x86_64.{transformed_id}"
    elif dataset_name.startswith("princeton-nlp/SWE-bench"):
        # SWE-bench format: docker.io/slimshetty/swebench-verified:sweb.eval.x86_64.{instance_id}
        image_name = f"docker.io/slimshetty/swebench-verified:sweb.eval.x86_64.{instance_id}"
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets are: {', '.join(sorted(SUPPORTED_DATASETS))}"
        )
    
    return image_name


@dataclass
class NanoConfig:
    agent_kind: str = "nano"
    model: Optional[str] = None
    api_base: str = "http://localhost:8000/v1"
    thinking: Optional[bool] = None  # None means omit parameter entirely
    token_limit: int = 8192
    tool_limit: int = 30
    time_limit: int = 60
    temperature: float = 0.7
    top_p: float = 0.95
    min_p: Optional[float] = None
    top_k: Optional[int] = None
    verbose: bool = False
    log: bool = False
    backend: str = "local"
    env: Optional[Any] = None


def _process_one(data: dict[str, Any], config: NanoConfig, dataset_name: Optional[str] = None) -> dict[str, Any]:
    assert "repo" in data and "base_commit" in data and "problem_statement" in data

    logger.info(f"[START] {data['repo']} @ {data['base_commit'][:7]}")
    start_time = time.time()

    agent_kwargs = asdict(config)
    agent_kwargs.pop("agent_kind", None)
    
    # Handle backend and environment setup
    backend = agent_kwargs.pop("backend", "local")
    env = agent_kwargs.pop("env", None)

    # Initialize agent with appropriate environment
    if backend == "apptainer" and env is None:
        # If using Apptainer backend but env not provided in config, we need to construct it
        # This logic might belong better in the caller, but we can handle it here or
        # expect the caller to pass the fully constructed environment in `config.env`.
        # Based on the reference, the caller (run_nano_eval) should likely construct the environment.
        # However, `_process_one` is called per instance, and the environment depends on the instance ID.
        instance_id = data.get("instance_id")
        if not instance_id:
            raise ValueError("instance_id is required when using apptainer backend")
        
        if not dataset_name:
            raise ValueError("dataset_name is required when using apptainer backend")
        
        image_name = _construct_image_name(instance_id, dataset_name)
        workdir = "/testbed"
        setup_fn = _get_setup_fn(dataset_name)
        env = ApptainerEnvironment(image=f"docker://{image_name}", workdir=workdir, setup_fn=setup_fn)
        agent_kwargs["env"] = env
    elif env:
        agent_kwargs["env"] = env

    agent = Agent(**agent_kwargs)

    diff = ""
    temp_folder = None
    
    if backend == "local":
        try:
            repo_url = handle_to_url(data["repo"])
            temp_folder = clone_repo_at_commit(repo_url, data["base_commit"])
        except Exception as e:
            agent._reset()
            agent._append({"role": "user", "content": data["problem_statement"]})
            agent._append({"role": "assistant", "content": ""})
            logger.error(f"Error with git in _process_one: {type(e).__name__}: {e}")
            if temp_folder: clean_repo_dir(temp_folder)
            return dict(
                prompt=agent.messages[:2],
                completion=agent.messages[2:],
                tools=agent.tools,
                generated_diff="",
                token_usage=agent.token_usage,
                tool_usage=agent.tool_usage,
                **agent.tool_stats
            )
    else:
        # For container backends, the repo is already in the container image at workdir
        temp_folder = "/testbed" # default workdir for SWE-bench containers

    try:
        diff = agent.run(task=data["problem_statement"], repo_root=temp_folder)
    except Exception as e:
        logger.error(f"Error in _process_one: {type(e).__name__}: {e}")
        diff = ""
    finally:
        if backend == "local" and temp_folder: clean_repo_dir(temp_folder)

        token_usage = agent.token_usage
        tool_usage = agent.tool_usage
        diff_success = diff != ""
        logger.info(f"[FINISH] {data['repo']} @ {data['base_commit'][:7]} - Tokens: {token_usage}, Tools: {tool_usage}, Diff Success: {diff_success}, Time: {time.time() - start_time:.2f}s")

    # Ensure agent.messages has enough messages to avoid empty completion
    if len(agent.messages) < 3:
        logger.warning(f"Agent messages incomplete ({len(agent.messages)} messages), padding with fallback")
        # Pad with user message and empty assistant response
        if len(agent.messages) < 2:
            agent._append({"role": "user", "content": data["problem_statement"]})
        if len(agent.messages) < 3:
            agent._append({"role": "assistant", "content": ""})
    
    result = dict(
        prompt=agent.messages[:2],
        completion=agent.messages[2:],
        tools=agent.tools,
        generated_diff=diff,
        token_usage=agent.token_usage,
        tool_usage=agent.tool_usage,
        **agent.tool_stats
    )
    return result


def nano_rollout_func(data: list[dict[str, Any]], config: NanoConfig, dataset_name: Optional[str] = None, **kwargs) -> list[dict[str, Any]]:
    """
    Deploys parallel Nano agents talking to our trl vllm-serve-async endpoint to process the given data.
    
    Args:
        data: List of data dictionaries, each containing instance information
        config: NanoConfig with agent configuration
        dataset_name: HuggingFace dataset name (e.g., "princeton-nlp/SWE-bench_Verified" or "SWE-Gym/SWE-Gym").
                     Required when using apptainer backend. Must be one of the supported datasets.
        **kwargs: Additional keyword arguments (ignored)
    
    Returns:
        List of result dictionaries, one per input data item
    
    Raises:
        ValueError: If dataset_name is not one of the supported datasets
    """
    # Validate dataset_name if provided or if using apptainer backend
    if dataset_name:
        if not (dataset_name.startswith("princeton-nlp/SWE-bench") or _is_swegym_dataset(dataset_name)):
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported datasets are: {', '.join(sorted(SUPPORTED_DATASETS))}"
            )
    elif config.backend == "apptainer":
        raise ValueError("dataset_name is required when using apptainer backend")

    logger.info(f"Starting {len(data)} agent rollouts" + (f" with dataset {dataset_name}" if dataset_name else ""))
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=min(len(data), os.cpu_count())) as executor:
        results = list(executor.map(lambda datum: _process_one(datum, config, dataset_name), data))

    logger.info(f"Finished {len(data)} rollouts in {time.time() - start_time:.2f}s")
    return results


if __name__ == "__main__":
    import time

    from src.data.swe_gym import get_swe_gym_repo_repair_dataset

    # Test different batch sizes for parallel timing
    batch_sizes = [2]
    runs = 1
    data = get_swe_gym_repo_repair_dataset().shuffle(seed=42)

    config = NanoConfig(model="hosted_vllm/Qwen/Qwen3-8B", backend="apptainer")

    avg_times = []

    for size in batch_sizes:
        print(f"Testing batch size {size}")
        subset = data.select(range(size))
        subset_dicts = [dict(x) for x in subset]
        times = []
        for i in range(runs):
            start_time = time.time()
            results = nano_rollout_func(subset_dicts, config, dataset_name="SWE-Gym/SWE-Gym")
            elapsed = time.time() - start_time
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}s")
        avg_time = sum(times) / runs
        avg_times.append(avg_time)
        print(f"Average time for batch size {size}: {avg_time:.2f}s\n")
