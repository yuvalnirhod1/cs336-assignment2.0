import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import timeit
from typing import Type, Any, Dict
from cs336_basics.transformer import Embeddings, Linear

class FSDP(torch.nn.Module):
    
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()

        for name, parameter in module.state_dict():
            module.state_dict
        for parameter in module.parameters():
            if isintance(parameter, Embeddings) or isintance(parameter, Linear):
                parameters.append(self.slice_parameter(parameter))
                del parameter
            else:
                parameters.append(parameter)
            
        self.layers = torch.nn.Sequential(*parameters)

    def construct_forward_hook(self, layer_index: int):
        def forward_hook(module: torch.nn.Module, inputs, output):
            pass

        return forward_hook

    def construct_backward_hook(self, layer_index: int):
        def backward_hook(module: torch.nn.Module, grad_inputs, grad_output):
            pass

        return backward_hook

    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        pass

class ShardedOptimizer(torch.optim.Optimizer):

    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        params = list(params)

        optimizer_params = self.slice_list_of_params(params)

        self.optimizer = optimizer_cls(optimizer_params, **kwargs)
        super().__init__(params, {})
    
    def update_lr(self, lr):
        self.optimizer.update_lr(lr)

    def slice_param(self, param):
        with torch.no_grad():
            parameters = param.view(-1)
            param_len = parameters.shape[0]
            return parameters[(param_len // self.world_size) * self.rank: (param_len // self.world_size) * (self.rank + 1)]

    def slice_list_of_params(self, param_list):
        new_param_list = []
        for param in param_list:
            new_param_list.append(self.slice_param(param))
        
        return new_param_list

    def synchronize_paramters(self):
        with torch.no_grad():
            for full_param_group, param_group in zip(self.param_groups, self.optimizer.param_groups):
                for full_param, param in zip(full_param_group["params"], param_group["params"]):
                    dist.all_gather_into_tensor(full_param.view(-1), param)
    
    def step(self, closure = None, **kwargs):
        for full_param_group, param_group in zip(self.param_groups, self.optimizer.param_groups):
            for full_param, param in zip(full_param_group["params"], param_group["params"]):
                if full_param.grad is not None:
                    chunk_size = full_param.numel() // self.world_size
                    param.grad = full_param.grad.detach().view(-1)[self.rank * chunk_size: (self.rank + 1) * chunk_size]
                
        self.optimizer.step(closure, **kwargs)
        self.synchronize_paramters()
    
    def add_param_group(self, param_group: Dict[str, Any]):
        optimizer_params = self.slice_list_of_params(param_group["params"])

        new_param_group = param_group.copy()
        new_param_group["params"] = optimizer_params 

        self.optimizer.add_param_group(new_param_group)
        super().add_param_group(param_group)

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, src_rank=0):
        super().__init__()
        self.module = module
        self.handles = []

        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=src_rank)
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(self.send_grad)
            
    def send_grad(self, param):
        with torch.no_grad():
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
            self.handles.append(handle)
    
    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, iteration_count, data_size):
    setup(rank, world_size)
    
    for _ in range(iteration_count):
        data = torch.rand((data_size,)).to(f"cuda:{rank}")
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize()

if __name__ == "__main__":
    world_size = 2

    iteration_count = int(sys.argv[1])
    data_size = int(sys.argv[2])

    print(f"Starting measurement: {iteration_count}, {data_size}")

    mp.spawn(fn=distributed_demo, args=(world_size, 5, data_size), nprocs=world_size, join=True)
    torch.cuda.synchronize()

    print("Finished warmup")

    start_time = timeit.default_timer()
    mp.spawn(fn=distributed_demo, args=(world_size, iteration_count, data_size), nprocs=world_size, join=True)
    overall_time = timeit.default_timer() - start_time

    print(f"Overall time: {overall_time}")