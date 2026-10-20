import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import timeit

WARMUP_ITERATIONS = 5

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, src_rank=0):
        super().__init__()
        self.module = module

        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=src_rank)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)
    
    def finish_gradient_synchronization(self):
        with torch.no_grad():
            for param in self.module.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=False)

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

    mp.spawn(fn=distributed_demo, args=(world_size, WARMUP_ITERATIONS, data_size), nprocs=world_size, join=True)
    torch.cuda.synchronize()

    print("Finished warmup")

    start_time = timeit.default_timer()
    mp.spawn(fn=distributed_demo, args=(world_size, iteration_count, data_size), nprocs=world_size, join=True)
    overall_time = timeit.default_timer() - start_time

    print(f"Overall time: {overall_time}")