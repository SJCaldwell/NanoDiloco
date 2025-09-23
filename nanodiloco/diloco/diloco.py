import torch
import torch.distributed as dist
import time
from transformers import get_cosine_schedule_with_warmup


class Diloco:
    def __init__(self, 
        model, 
        inner_optimizer, 
        outer_optimizer, 
        warmup_steps, 
        total_steps,
        inner_steps: int = 100, 
        outer_steps: int = 10
    ):
        self.model = model
        self.inner_optimizer = inner_optimizer
        self.outer_optimizer = outer_optimizer
        self.scheduler = get_cosine_schedule_with_warmup(self.inner_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        for param in self.model.parameters():
            rank_0_param = param.data.clone()
            dist.broadcast(rank_0_param, src=0)
        self._sync_time = 0
        self._sync_calls = 0
        self.offloaded_last_sync_parameters = self._get_offloaded_parameters()
    
    def _get_offloaded_parameters(self):
        return [
            param.data.detach().clone().to("cpu")
            for group in self.outer_optimizer.param_groups
            for param in group["params"]
        ]
    
    def outer_step(self) -> None:
        """
        Outer step for Diloco.
        Loads last sync parameters from CPU to GPU and computes the psuedo-gradient for outer optimizer.
        Updates the offloaded parameters to CPU.
        """
        replica_params = [
            param
            for group in self.inner_optimizer.param_groups
            for param in group["params"]
        ]

        for replica_param, last_sync_param in zip(replica_params, self.offloaded_last_sync_parameters):
            last_sync_param_on_device = last_sync_param.to(replica_param.device)
            last_sync_param.grad = last_sync_param_on_device - replica_param.data
            dist.all_reduce(tensor=last_sync_param.grad, op=dist.ReduceOp.AVG)
            replica_param.data = last_sync_param_on_device
        
        self.outer_optimizer.step()
        self.outer_optimizer.zero_grad()
        self.offloaded_last_sync_parameters = self._get_offloaded_parameters()

    def inner_step(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # gradient clipping
        self.inner_optimizer.step()
        self.scheduler.step()
        self.inner_optimizer.zero_grad()

    @property
    def avg_sync_time(self):
        return self._sync_time / self._sync_calls if self._sync_calls > 0 else 0
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()