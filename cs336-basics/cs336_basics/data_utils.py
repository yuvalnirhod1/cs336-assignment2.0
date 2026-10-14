from typing import List, Tuple, Dict
import numpy as np
import torch
from einops import rearrange


def get_batch(x: np.array, batch_size: int, context_length: int, device: torch.device):
    assert len(x) > 1, "No way we can do next token prediction with only 1 token"

    x = torch.tensor(x, device=device)

    inputs = torch.zeros((batch_size, context_length), device=device)
    targets = torch.zeros((batch_size, context_length), device=device)

    assert batch_size - 1 + context_length < len(x), "Batch size too big"

    batch_indices = torch.randint(0, len(x) - context_length, size=(batch_size,))
    for index, batch_index in enumerate(batch_indices):
        inputs[index] = x[batch_index:batch_index + context_length]
        targets[index] = x[batch_index + 1:batch_index + context_length + 1]

    return inputs.long(), targets.long()

class Checkpointing:
    def __init__(self):
        raise NotImplemented

    @staticmethod
    def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str):
        obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
        torch.save(obj, out)

    @staticmethod
    def load_checkpoint(src: str, model: torch.nn.Module = None, optimizer: torch.optim.Optimizer = None):
        obj = torch.load(src, weights_only=False)
        if model:
            model.load_state_dict(obj["model"])
        if optimizer:
            optimizer.load_state_dict(obj["optimizer"])
        return obj["iteration"]

def main():
    pass

if __name__ == "__main__":
    main()