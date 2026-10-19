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

class OperationMode(IntEnum):
    FORWARD = 0
    FORWARD_BACKWORD = 1
    FORWARD_BACKWORD_STEP = 2

def measure_training_loop(model: torch.nn.Module, warmup_count: int, iteration_count: int, operation_mode: int, vocab_size: int, batch_size: int, context_length: int, device: torch.device, autocast: bool) -> float:
    ### Get user dataset input
    optimizer_config = DEFAULT_CONFIG["optimizer"]
    training_config = DEFAULT_CONFIG["training"]
    ### Configure optimizer
    lr = optimizer_config["lr"] # 
    optimizer = AdamW(model.parameters(),
                        lr,
                        optimizer_config["weight_decay"],
                        betas=optimizer_config["betas"],
                        eps=optimizer_config["eps"])

    ### Training loop
    total_time = 0
    for iteration in tqdm(range(warmup_count + iteration_count)):

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if autocast else nullcontext():
            measure = iteration >= warmup_count
                
            if measure:
                start_time = timeit.default_timer()

            data = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
            labels = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

            if operation_mode >= OperationMode.FORWARD:
                with (nvtx.range("Forward pass") if measure else nullcontext()):
                    guessed_labels = model(data)

            guessed_labels = rearrange(guessed_labels, "batch_size sequence_length vocab -> (batch_size sequence_length) vocab")
            labels = rearrange(labels, "batch_size sequence_length -> (batch_size sequence_length)")

            if operation_mode >= OperationMode.FORWARD_BACKWORD:
                with nvtx.range("Backward pass") if measure else nullcontext():
                    batch_loss = cross_entropy(guessed_labels, labels)
                    batch_loss.backward()
                    gradient_clipping(model.parameters(), optimizer_config["max_l2_norm"])

                del batch_loss

            lr = calculate_cosine_lr(iteration,
                                optimizer_config["lr"],
                                optimizer_config["lr"] / 10,
                                training_config["number_of_iterations"] * 0.05,
                                training_config["number_of_iterations"])

            if operation_mode >= OperationMode.FORWARD_BACKWORD_STEP:
                with nvtx.range("Optimization step") if measure else nullcontext():
                    optimizer.update_lr(lr)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if measure:
                if torch.device == torch.cuda:
                    torch.cuda.synchronize()
                end_time = timeit.default_timer()
                total_time += (end_time - start_time)

        del data, labels, guessed_labels

    return total_time / iteration_count

OPERATION_MODES = {
    "forward": 0,
    "backward": 1,
    "step": 2
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup_count", type=int, default=5)
    parser.add_argument("--iteration_count", type=int, default=10)
    parser.add_argument("--operation_mode", type=str, default="step")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--autocast", action="store_true", default=False)
    parser.add_argument("--memory_measure", action="store_true", default=False)
    parser.add_argument("--checkpoints", type=int, default=None)
    return parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    args = get_args()

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

    transformer = torch.compile(transformer)

    operation_mode = OPERATION_MODES[args.operation_mode]

    if args.memory_measure:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    total_time = measure_training_loop(transformer, args.warmup_count, args.iteration_count, operation_mode, args.vocab_size, args.batch_size, args.context_length, device, args.autocast)
    print(f"Average iteration time {total_time}")

    if args.memory_measure:
        torch.cuda.memory._dump_snapshot("memory_usage.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

if __name__ == "__main__":
    main()
