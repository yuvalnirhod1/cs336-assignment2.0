import torch
from einops import rearrange
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np

if __name__ == "__main__":
    from transformer import softmax
else:
    from .transformer import softmax

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    inputs = inputs - torch.amax(inputs, dim=-1, keepdim=True)
    summation = torch.log(torch.sum(torch.exp(inputs), dim=-1, keepdim=True))
    value = torch.take_along_dim(summation - inputs, rearrange(targets, "... index -> ... index 1"), dim=-1)
    return value.mean()

def perplexity(cross_entropy_loss: torch.Tensor) -> torch.Tensor:
    return torch.exp(cross_entropy_loss)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        assert lr >= 0, "LR must be positive!"
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    def update_lr(self, lr):
        for group in self.param_groups:
            group["lr"] = lr

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad = p.grad.data

                t = state.get("t", 1)
                m = state.get("m", 0)
                v = state.get("v", 0)

                lr_t = lr * np.sqrt(1 - np.pow(beta2, t)) / (1 - np.pow(beta1, t))
                p.data = p.data - lr * weight_decay * p.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.pow(grad, 2)

                p.data = p.data - lr_t * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss

def calculate_cosine_lr(t, lr_max, lr_min, warmup_time, cosine_time) -> torch.float32:
    if t < warmup_time:
        return (t / warmup_time) * lr_max
    if t <= cosine_time:
        return lr_min + (1 + np.cos(np.pi * (t - warmup_time) / (cosine_time - warmup_time))) * (lr_max - lr_min) / 2
    return lr_min

def gradient_clipping(params: List[torch.nn.Parameter], max_l2_norm: torch.float32, eps=1e-6):
    parameters = [p for p in params if p.grad is not None]
    grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach(), 2.0) for p in parameters]), 2.0)

    if grad_norm > max_l2_norm:
        dim = (max_l2_norm / (grad_norm + eps))
        for p in parameters:
            p.grad *= dim