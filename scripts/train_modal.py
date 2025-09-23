import os
from pathlib import Path

import modal
import modal.experimental

# CUDA configuration
cuda_version = "12.6.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Paths
LOCAL_CODE_DIR = Path(__file__).parent.parent.absolute()
REMOTE_CODE_DIR = "/root/nanodiloco"
REMOTE_TRAIN_SCRIPT = "/root/nanodiloco/main.py"
GPU_TYPE = "T4"
VOLUME_NAME = "hf-c4-tiny"

# Cluster configuration
n_nodes = 2
n_proc_per_node = 1  # 1 GPUs per node for multi-node (2 total)
single_node_gpus = 2  # 2 GPUs for single node training

# Base image with dependencies
base_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.13")
    .apt_install("git", "libibverbs-dev", "libibverbs1")
    # Install runtime deps from the project's pyproject.toml
    .pip_install_from_pyproject(str(LOCAL_CODE_DIR / "pyproject.toml"))
)

# Add local code to image
image = base_image.add_local_dir(
    LOCAL_CODE_DIR,
    remote_path="/root",
)

app = modal.App("nanodiloco-training", image=image)

# Modal volumes
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
volume_model_output = modal.Volume.from_name(
    "nanodiloco-model-output", create_if_missing=True
)


def _train_single_node(
    batch_size: int = 256,
    lr: float = 4e-4,
    steps: int = 10000,
    llama_config_path: str = None,
    wandb_config_path: str = None,
):
    """Helper function to run training on a single node."""
    from torch.distributed.run import parse_args, run

    # Build arguments for nanodiloco training
    args = [
        f"--nproc-per-node={single_node_gpus}",
        "-m", "nanodiloco.main",
        f"--batch-size={batch_size}",
        f"--lr={lr}",
        f"--steps={steps}",
        "--dataset-path=/vol/datasets/PrimeIntellect/c4-tiny/en/save_to_disk",
    ]

    if llama_config_path:
        args.extend([f"--llama-config-file={llama_config_path}"])
    if wandb_config_path:
        args.extend([f"--wandb-config-file={wandb_config_path}"])

    print(f"Running torchrun with args: {' '.join(args)}")
    run(parse_args(args))


@app.function(
    gpu=f"T4:{single_node_gpus}",
    secrets=[
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={
        "/vol": volume,
        "/root/out": volume_model_output,
    },
    timeout=60 * 60 * 4,  # 4 hours
)
def train_single_node(
    batch_size: int = 256,
    lr: float = 4e-4,
    steps: int = 10000,
    llama_config_path: str = None,
    wandb_config_path: str = None,
):
    """Train the model on a single node with N GPUs."""
    _train_single_node(batch_size, lr, steps, llama_config_path, wandb_config_path)


def _train_multi_node(
    batch_size: int = 512,  # Larger default for multi-node
    lr: float = 4e-4,
    steps: int = 10000,
    llama_config_path: str = None,
    wandb_config_path: str = None,
) -> None:
    """Helper function to run multi-node distributed training."""
    from torch.distributed.run import parse_args, run

    cluster_info = modal.experimental.get_cluster_info()
    container_rank: int = cluster_info.rank
    main_ip_addr: str = cluster_info.container_ips[0]
    container_id = os.environ["MODAL_TASK_ID"]

    print(f"Hello from {container_id}, rank {container_rank} of {n_nodes}")
    if container_rank == 0:
        print(f"Main container's address: {main_ip_addr}")

    # Build arguments for distributed training
    args = [
        f"--nnodes={n_nodes}",
        f"--nproc-per-node={n_proc_per_node}",
        f"--node-rank={cluster_info.rank}",
        f"--master-addr={main_ip_addr}",
        "-m", "nanodiloco.main",
        f"--batch-size={batch_size}",
        f"--lr={lr}",
        f"--steps={steps}",
        "--dataset-path=/vol/datasets/PrimeIntellect/c4-tiny/en/save_to_disk",
    ]

    if llama_config_path:
        args.extend([f"--llama-config-file={llama_config_path}"])
    if wandb_config_path:
        args.extend([f"--wandb-config-file={wandb_config_path}"])

    print(f"Running torchrun with args: {' '.join(args)}")
    run(parse_args(args))


@app.function(
    gpu=f"T4:{n_proc_per_node}",
    secrets=[
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={
        "/vol": volume,
        "/root/out": volume_model_output,
    },
    timeout=60 * 60 * 8,  # 8 hours for multi-node
    cloud="auto",
)
@modal.experimental.clustered(n_nodes, rdma=False)
def train_multi_node(
    batch_size: int = 512,
    lr: float = 4e-4,
    steps: int = 10000,
    llama_config_path: str = None,
    wandb_config_path: str = None,
):
    """Train the model on a multi-node cluster with N GPUs per node."""
    _train_multi_node(batch_size, lr, steps, llama_config_path, wandb_config_path)


@app.function(
    gpu=f"T4:{n_proc_per_node}",
    volumes={
        "/vol": volume,
        "/root/out": volume_model_output,
    },
    timeout=60 * 30,  # 30 minutes for benchmark
    cloud="auto",
)
@modal.experimental.clustered(n_nodes, rdma=False)
def benchmark_multi_node(
    batch_size: int = 512,
    steps: int = 200,  # Short benchmark run
):
    """Run a benchmark of multi-node training performance."""
    # Set environment variables for benchmarking
    os.environ["NANODILOCO_BENCHMARK"] = "true"
    _train_multi_node(batch_size=batch_size, steps=steps)


@app.function(
    cpu=4.0,
    volumes={"/vol": volume},
    timeout=60 * 60,  # 1 hour
)
def prepare_configs():
    """Prepare default configuration files on the volume."""
    import json

    # Default Llama config from your specification
    llama_config = {
        "architectures": [
            "LlamaForCausalLM"
        ],
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 6,
        "rms_norm_eps": 1e-05,
        "use_cache": False
    }

    # Default W&B config from your specification
    wandb_config = {
        "nodes": n_nodes,
        "location": "modal",
        "backend": "nccl",
        "measure_comms": True
    }

    # Large model config for scaling experiments
    large_llama_config = {
        "architectures": [
            "LlamaForCausalLM"
        ],
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_attention_heads": 8,
        "num_hidden_layers": 12,
        "rms_norm_eps": 1e-05,
        "use_cache": False
    }

    # Create configs directory
    config_dir = Path("/vol/configs")
    config_dir.mkdir(exist_ok=True)

    # Write config files
    configs = [
        ("llama_default.json", llama_config),
        ("llama_large.json", large_llama_config),
        ("wandb_default.json", wandb_config),
    ]

    for filename, config in configs:
        config_path = config_dir / filename
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created config: {config_path}")


# Convenience functions for common training scenarios
@app.local_entrypoint()
def small_single_node():
    """Run small model training on single node."""
    train_single_node.remote(
        batch_size=128,
        lr=1e-3,
        steps=5000,
        llama_config_path="/root/configs/llama_default.json",
        wandb_config_path="/root/configs/wandb_default.json"
    )


@app.local_entrypoint()
def large_multi_node():
    """Run large model training on multi-node cluster."""
    train_multi_node.remote(
        batch_size=1024,
        lr=4e-4,
        steps=10000,
        llama_config_path="/root/configs/llama_default.json",
        wandb_config_path="/root/configs/wandb_default.json"
    )


@app.local_entrypoint()
def benchmark():
    """Run a quick benchmark of multi-node performance."""
    benchmark_multi_node.remote()


@app.local_entrypoint()
def main():
    """Default: run medium-scale multi-node training."""
    train_multi_node.remote(
        llama_config_path="/root/configs/llama_default.json",
        wandb_config_path="/root/configs/wandb_default.json"
    )


if __name__ == "__main__":
    main()