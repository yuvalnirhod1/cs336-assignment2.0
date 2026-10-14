from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple
import os

def create_logger(base_dir: str, config: Dict):
    base_dir = base_dir
    subfolder_name = "__".join([f"{k}:v" for k, v in config.values()])
    log_dir = os.path.join(base_dir, subfolder_name)
    return SummaryWriter(log_dir)