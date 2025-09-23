import torch
import torch.distributed as dist
import time



class Diloco:
    def __init__(self, model, inner_optimizer, outer_optimizer, inner_steps: int = 100, outer_steps: int = 10):
        self.model = model
        self.inner_optimizer = inner_optimizer
        self.outer_optimizer = outer_optimizer
        for param in self.model.parameters():
            rank_0_param = param.data.clone()
            dist.broadcast(rank_0_param, src=0)
        self._sync_time = 0
        self._sync_calls = 0
        self.offloaded_parameters = self._get_offloaded_parameters()
    
    def _get_offloaded_parameters(self):
        return [
            param.data.detach().clone().to("cpu")
            for group in self.outer_optimizer.param_groups
            for param in group["params"]
        ]
    
    