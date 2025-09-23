from diloco import Diloco
from training_utils import ddp_setup, create_run_name, get_tokenized_dataset, get_tokenizer, get_tokenized_dataset, set_seed_all
from cyclopts import App
import os
import torch.distributed as dist
import wandb
import torch
from typing import Optional
from transformers import LlamaConfig, LlamaForCausalLM
from torch.utils.data import DataLoader
from datasets.distributed import split_dataset_by_node
import json

app = App("nanodiloco")

def get_default_llama_config():
    return {
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

def get_default_wandb_config():
    return {
        "nodes": 1,
        "location": "local",
        "backend": "nccl",
        "measure_comms": True
    }

def load_config_from_file(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)

@app.default
def train_model(
    seed: int = 1337,
    batch_size: int = 256,
    per_device_batch_size: int = 8,
    seq_length: int = 1024,
    warmup_steps: int = 100,
    total_steps: int = 10_000,
    inner_steps: int = 100,
    lr: float = 4e-4,
    outer_lr: float = 0.7,
    project: str = "nano-diloco",
    dataset_path: str = "/mnt/hf-c4-tiny/datasets/PrimeIntellect/c4-tiny/en/save_to_disk",
    llama_config_file: Optional[str] = None,
    wandb_config_file: Optional[str] = None,
) -> None:
    llama_config = load_config_from_file(llama_config_file) if llama_config_file else get_default_llama_config()
    wandb_config = load_config_from_file(wandb_config_file) if wandb_config_file else get_default_wandb_config()

    set_seed_all(seed)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])

    assert batch_size % per_device_batch_size == 0 # check even gradient accumulation
    gradient_accumulation_steps = batch_size / per_device_batch_size

    outer_steps = total_steps // inner_steps
    assert total_steps % inner_steps == 0 # check even outer steps

    if global_rank == 0:
        run_name = create_run_name(experiment_type="nanodiloco", node_config=wandb_config, is_debug=False)
        wandb.init(project=project, name=run_name, config=wandb_config)
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = get_tokenizer()
    tokenized_ds = get_tokenized_dataset(dataset_path, tokenizer)
    train_dataset = split_dataset_by_node(tokenized_ds, world_size=world_size, rank=global_rank)

    def collate_fn(batch):
        padded = tokenizer.pad(
            batch,
            padding="longest",
            max_length=seq_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        padded['labels'] = padded['input_ids'].clone()
        return padded
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True
    )
    config = LlamaConfig(**llama_config)
    model = LlamaForCausalLM(config)
    model.to(local_rank) #type: ignore
    inner_optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    outer_optimizer = torch.optim.SGD(model.parameters(), lr=outer_lr, momentum=0.9, nesterov=True)
    diloco_model = Diloco(model, inner_optimizer, outer_optimizer, warmup_steps, total_steps, inner_steps=inner_steps, outer_steps=outer_steps)

    diloco_model.train()

    for (i, batch) in enumerate(train_dataloader):
        real_step = (i + 1) // gradient_accumulation_steps
        batch = {k: v.to(device) for k, v in batch.items()}
        output = diloco_model(**batch)
        loss = output.loss / gradient_accumulation_steps
        output.loss.backward()
        if (i + 1) % gradient_accumulation_steps == 0:
            diloco_model.inner_step()

            if real_step % inner_steps == 0:
                diloco_model.outer_step()
        
        if local_rank == 0:
            dict_to_log = {
                "loss": loss.item(),
                "step": real_step,
                "lr": [group["lr"] for group in diloco_model.inner_optimizer.param_groups][0],
                "Perplexity": torch.exp(loss).item(),
                "effective_step": real_step * world_size,
                "total_samples": real_step * batch_size * world_size
            }
            wandb.log(dict_to_log)
    
    print("Training completed!")
    wandb.finish()


def main():
    print("Training Diloco with NanoDiloco...")
    ddp_setup()
    app()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
