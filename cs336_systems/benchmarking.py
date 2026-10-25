import cs336_basics.transformer as model
import cs336_basics.data_utils as data_utils
from cs336_basics.loss import cross_entropy, AdamW, gradient_clipping, calculate_cosine_lr
import argparse
import torch
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from einops import rearrange
from cs336_basics.config import DEFAULT_CONFIG
from enum import IntEnum
import timeit
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext
import torch.multiprocessing as mp
import torch.distributed as dist
from .multiprocess import DDP, FSDP, ShardedOptimizer
import os

def _cleanup_process_group():
    # Synchronize before we destroy the process group
    dist.barrier()
    dist.destroy_process_group()

def _setup_process_group(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    # https://discuss.pytorch.org/t/should-local-rank-be-equal-to-torch-cuda-current-device/150873/2
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device

def measure_training_loop(model: torch.nn.Module, iteration_count: int, vocab_size: int, batch_size: int, context_length: int, device: torch.device, rank: int, world_size: int) -> float:
    ### Get user dataset input
    optimizer_config = DEFAULT_CONFIG["optimizer"]
    training_config = DEFAULT_CONFIG["training"]
    ### Configure optimizer
    lr = optimizer_config["lr"] # 
    optimizer = ShardedOptimizer(model.parameters(), AdamW,
                        lr=lr,
                        weight_decay=optimizer_config["weight_decay"],
                        betas=optimizer_config["betas"],
                        eps=optimizer_config["eps"])

    # optimizer = AdamW(model.parameters(),
    #                     lr=lr,
    #                     weight_decay=optimizer_config["weight_decay"],
    #                     betas=optimizer_config["betas"],
    #                     eps=optimizer_config["eps"])

    ### Training loop
    total_time = 0

    assert batch_size % world_size == 0, "world size must divide batch size"

    for iteration in tqdm(range(iteration_count)):

        data = torch.randint(0, vocab_size, (batch_size // world_size, context_length), device=device)
        labels = torch.randint(0, vocab_size, (batch_size // world_size, context_length), device=device)

        guessed_labels = model(data)

        guessed_labels = rearrange(guessed_labels, "batch_size sequence_length vocab -> (batch_size sequence_length) vocab")
        labels = rearrange(labels, "batch_size sequence_length -> (batch_size sequence_length)")

        batch_loss = cross_entropy(guessed_labels, labels)
        batch_loss.backward()
        gradient_clipping(model.parameters(), optimizer_config["max_l2_norm"])

        del batch_loss

        lr = calculate_cosine_lr(iteration,
                            optimizer_config["lr"],
                            optimizer_config["lr"] / 10,
                            training_config["number_of_iterations"] * 0.05,
                            training_config["number_of_iterations"])

        optimizer.update_lr(lr)

        try:
            model.finish_gradient_synchronization()
        except:
            pass

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        del data, labels, guessed_labels

    return total_time / iteration_count

OPERATION_MODES = {
    "forward": 0,
    "backward": 1,
    "step": 2
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_count", type=int, default=20)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=1280)
    parser.add_argument("--num_heads", type=int, default=20)
    parser.add_argument("--num_layers", type=int, default=36)
    parser.add_argument("--d_ff", type=int, default=5120)
    parser.add_argument("--memory_measure", action="store_true", default=False)
    parser.add_argument("--checkpoints", type=int, default=None)
    return parser.parse_args()

def main():
    world_size = 2

    args = get_args()

    start_time = timeit.default_timer()

    mp.spawn(
        distributed_training,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )
    
    end_time = timeit.default_timer()

    print(f"Total time: {end_time - start_time}")

def distributed_training(rank, world_size, args):
    device = _setup_process_group(rank, world_size, "nccl")
    print(f"Device is {device}")

    if args.memory_measure:
        print("Recording memory usage")
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    transformer = model.TransformerLM(
        args.d_model,
        args.num_heads,
        args.vocab_size,
        args.context_length,
        args.num_layers,
        args.d_ff,
        device=device,
        checkpoints=args.checkpoints
    )

    measure_training_loop(DDP(transformer), args.iteration_count, args.vocab_size, args.batch_size, args.context_length, device, rank, world_size)

    if args.memory_measure:
        print("Saving memory usage")
        torch.cuda.memory._dump_snapshot("memory_usage.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    _cleanup_process_group()


if __name__ == "__main__":
    main()
