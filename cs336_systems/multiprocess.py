import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import timeit
from typing import Type, Any, Dict

import cs336_basics.model 
import cs336_basics.transformer 
import copy

def alloc_storage(tensor: torch.Tensor, size: int):
    storage = tensor.untyped_storage()
    if storage.size() != size:
        storage.resize_(size)

def free_storage(tensor: torch.Tensor):
    alloc_storage(tensor, 0)

def revive_storage(tensor: torch.Tensor):
    alloc_storage(tensor, tensor.nbytes)

def master_print(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)

LOOKAHEAD_COUNT = 2

class FSDP(torch.nn.Module):
    # TODO:
    # 1. Add compute_dtype usage

    SHARD_TYPES = (cs336_basics.model.Embedding,
                   cs336_basics.model.Linear,
                   cs336_basics.transformer.Embeddings,
                   cs336_basics.transformer.Linear)
    BACKWARD_INDEX_RESET_VALUE = -1
    
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.end_handles = []

        self.backward_current_index = self.BACKWARD_INDEX_RESET_VALUE

        self.lookahead_count = 0
        self.index_to_layer = {}
        self.max_index = 0

        self.atomic_layers = [(name, module) for name, module in module.named_modules() if len(list(module.children())) == 0]

        saved_parameters = torch.nn.ParameterDict()
        for name, sub_module in self.atomic_layers:
            if isinstance(sub_module, self.SHARD_TYPES):
                saved_parameters[name] = self.split_module(sub_module)

                sub_module.register_forward_pre_hook(self.construct_pre_forward_hook(name))
                sub_module.register_forward_hook(self.construct_post_forward_hook(name, True))
                sub_module.register_full_backward_pre_hook(self.construct_pre_backward_hook(name))
                sub_module.register_full_backward_hook(self.construct_post_backward_hook(name))

                for param_name, parameter in sub_module.named_parameters():
                    parameter.register_post_accumulate_grad_hook(self.construct_post_grad_hook(name, param_name))
            else:
                sub_module.register_forward_hook(self.construct_post_forward_hook(name, False))
                sub_module.register_full_backward_hook(self.construct_post_backward_hook(name))

                for param_name, parameter in sub_module.named_parameters():
                    if parameter.requires_grad:
                        parameter.register_post_accumulate_grad_hook(self.send_grad)
        
        self.saved_parameters = saved_parameters
        self.module = module
    
    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True,):
        for name, parameter in self.module.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate):
            chosen_parameter = parameter
            for module_name, parameters in self.saved_parameters.items():
                for shraded_parameter_name, shraded_parameter in parameters.items():
                    if f"{module_name}.{shraded_parameter_name}" == name:
                        chosen_parameter = shraded_parameter
            yield name, chosen_parameter 

    def parameters(self, recurse: bool = True):
        for _, param in self.named_parameters(recurse=recurse):
            yield param
            
    def slice_tensor(self, tensor):
        return torch.nn.Parameter(tensor.view(-1).chunk(self.world_size)[self.rank].clone())
        
    @torch.no_grad()
    def split_module(self, module):
        sliced_parameters = {}
        for name, parameter in module.named_parameters():
            sliced_parameters[name] = self.slice_tensor(parameter)
        self.free_module(module)
        return sliced_parameters

    @torch.no_grad() 
    def all_gather_module(self, module, module_name, dtype=None):
        handles = []
        if not dtype:
            dtype = self.compute_dtype

        for name, parameter in module.named_parameters():
            revive_storage(parameter)
            shraded_parameter = self.saved_parameters[module_name][name]
            parameter.data = parameter.data.to(dtype)
            handles.append(dist.all_gather_into_tensor(parameter.data.view(-1), shraded_parameter.data.to(dtype), async_op=True))
        return handles

    @torch.no_grad()
    def free_module(self, module):
        for parameter in module.parameters():
            parameter.grad = None
            free_storage(parameter) 

    def _is_layer_sharded(self, index):
        if index >= self.max_index or index < 0:
            return False
        lookahead_layer, lookahead_layer_name = self.index_to_layer[index]
        return lookahead_layer_name in self.saved_parameters
    
    def construct_pre_forward_hook(self, layer_name: str):
        def forward_hook(module: torch.nn.Module, inputs):
            if self.lookahead_count == 0:
                self.sharded_handles[self.current_index] = self.all_gather_module(module, layer_name)

            for handle in self.sharded_handles[self.current_index]:
                handle.wait()
            del self.sharded_handles[self.current_index]

        return forward_hook

    def construct_post_forward_hook(self, layer_name: str, should_free: bool):
        def forward_hook(module: torch.nn.Module, inputs, output):
            if should_free:
                self.free_module(module)

            if self.lookahead_count == 0:
                self.index_to_layer[self.current_index] = (module, layer_name)
            else:
                index = self.current_index + self.lookahead_count
                if self._is_layer_sharded(index):
                    self.sharded_handles[index] = self.all_gather_module(*self.index_to_layer[index])

            self.current_index += 1
            
        return forward_hook
    
    def _setup_backward_pass(self):
        assert self.max_index > 0, "Forward was not called yet! (Or no layers to model)"
        self.backward_current_index = self.max_index - 1
        self.backward_sharded_handles = {}

    def construct_pre_backward_hook(self, layer_name: str):
        def backward_hook(module: torch.nn.Module, grad_inputs):
            if self.backward_current_index == self.BACKWARD_INDEX_RESET_VALUE:
                self._setup_backward_pass()
                self.backward_sharded_handles[self.backward_current_index] = self.all_gather_module(module, layer_name)

            for handle in self.backward_sharded_handles[self.backward_current_index]:
                handle.wait()
            del self.backward_sharded_handles[self.backward_current_index]

        return backward_hook

    def construct_post_backward_hook(self, layer_name: str):
        def backward_hook(module: torch.nn.Module, grad_inputs, grad_outputs):
            # We need this both in pre and post hook to handle the case where the last layer is 
            # not pre-hooked. In the forward passed we managed to escape this by using forward
            if self.backward_current_index == self.BACKWARD_INDEX_RESET_VALUE:
                self._setup_backward_pass()

            if self.backward_current_index == self.max_index - 1 and self.lookahead_count > 0:
                for i in range(self.lookahead_count - 1):
                    index = self.backward_current_index - i - 1
                    _, layer_name = self.index_to_layer[index]
                    if self._is_layer_sharded(index):
                        self.backward_sharded_handles[index] = self.all_gather_module(*self.index_to_layer[index])

            index = self.backward_current_index - self.lookahead_count
            if self._is_layer_sharded(index):
                _, layer_name = self.index_to_layer[index]
                self.backward_sharded_handles[index] = self.all_gather_module(*self.index_to_layer[index])

            self.backward_current_index -= 1

        return backward_hook

    def construct_post_grad_hook(self, layer_name: str, name: str):
        def backward_hook(parameter):
            with torch.no_grad():
                sliced_gradient = torch.empty_like(self.saved_parameters[layer_name][name])
                handle = dist.reduce_scatter_tensor(sliced_gradient, parameter.grad.view(-1).to(torch.float32), op=dist.ReduceOp.AVG, async_op=True)

                if self.saved_parameters[layer_name][name].grad is None:
                    self.saved_parameters[layer_name][name].grad = sliced_gradient
                else:
                    self.saved_parameters[layer_name][name].grad += sliced_gradient

                parameter.grad = None
                free_storage(parameter)
                self.end_handles.append(handle)
        return backward_hook

    def gather_full_params(self):

        handles = []

        for submodule_name, sub_module in self.atomic_layers:
            if isinstance(sub_module, self.SHARD_TYPES):
                handles.extend(self.all_gather_module(sub_module, submodule_name, dtype=torch.float32))
        
        for handle in handles:
            handle.wait()

        new_state_dict = copy.deepcopy(self.module.state_dict())

        for submodule_name, sub_module in self.atomic_layers:
            if isinstance(sub_module, self.SHARD_TYPES):
                self.free_module(sub_module)
        
        return new_state_dict

    def send_grad(self, param):
        with torch.no_grad():
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
            self.end_handles.append(handle)
    
    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        self.current_index = 0
        self.sharded_handles = {}

        for i in range(self.lookahead_count):
            lookahead_layer, lookahead_layer_name = self.index_to_layer[i]
            if lookahead_layer_name in self.saved_parameters:
                self.sharded_handles[i] = self.all_gather_module(lookahead_layer, lookahead_layer_name)

        output = self.module(*inputs, **kwargs)

        if self.lookahead_count == 0:
            self.max_index = self.current_index
            self.lookahead_count = LOOKAHEAD_COUNT
        
        return output
    
    def finish_gradient_synchronization(self):
        for handle in self.end_handles:
            handle.wait()
        self.end_handles.clear()

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