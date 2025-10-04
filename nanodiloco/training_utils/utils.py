import random
from datasets import load_from_disk # type: ignore
import numpy as np
import torch
from datetime import datetime
import uuid
import torch.distributed as dist
import os
from transformers import AutoTokenizer

def set_seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_run_name(experiment_type, node_config, is_debug=False) -> str:
    """
    experiment_type: 'baseline', 'single-node', 'multi-node', 'geo-distributed'
    node_config: dict with node setup details
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    # Build hierarchical name
    parts = [experiment_type]
    
    if node_config.get('nodes'):
        parts.append(f"n{node_config['nodes']}")
    if node_config.get('location'):
        parts.append(node_config['location'])
    
    parts.append(timestamp)
    
    if is_debug:
        parts.insert(0, "debug")
    
    base_experiment_name = "_".join(parts)
    return f"{base_experiment_name}_{str(uuid.uuid4())[:8]}"

def ddp_setup() -> None:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def get_tokenized_dataset(dataset_path, tokenizer, seq_length=1024):
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    dataset = load_from_disk(dataset_path)

    def tokenize_function(data):
        outputs = tokenizer(data["text"], truncation=True, max_length=seq_length)
        return outputs
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "timestamp", "url"]
    )
    return tokenized_datasets["train"]

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.pad_token = "</s>"
    return tokenizer