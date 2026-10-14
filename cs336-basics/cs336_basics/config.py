from typing import Dict
import torch

DEFAULT_CONFIG = {
    "model": {
        "d_model": 512,
        "num_heads": 16,
        "vocab_size": 10000,
        "context_length": 128,
        "num_layers": 4,
        "d_ff": 1344,
        "theta": 10000,
        "dtype": torch.float
    },
    "optimizer": {
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "max_l2_norm": 0.2,
        "lr": 1e-3
    },
    "training": {
        "number_of_iterations": 1000,
        "batch_size": 128,
        "save_iteration_number": 5
    }
}

def update_config(user_config: Dict = None) -> Dict:
    config = DEFAULT_CONFIG

    if user_config is not None: 
        for module in config.keys():
            for k, v in user_config[module]:
                config[module][k] = v

    return config
