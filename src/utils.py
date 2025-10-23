# utils.py
from pathlib import Path
import random
from typing import Optional
import numpy as np
import torch
from torchvision.utils import save_image, make_grid


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_sample_grid(tensors: torch.Tensor, out_dir: str, file_name: str, nrow: int = 8):
    """Saves a grid of images in [−1,1] to disk as PNG in [0,1]."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    grid = make_grid((tensors + 1) / 2, nrow=nrow, pad_value=1.0)
    save_image(grid, str(Path(out_dir) / file_name))
