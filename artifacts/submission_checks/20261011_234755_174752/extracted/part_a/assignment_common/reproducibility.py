"""Seed setup and loaders that keep training and evaluation RNGs separate."""

from __future__ import annotations

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_everything(seed: int, *, deterministic: bool = True, warn_only: bool = True) -> None:
    """Call before constructing a model, especially before first CUDA use."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(dataset, batch_size: int, *, training: bool = False, seed: int = 42,
                num_workers: int = 0, device: str = "cpu") -> DataLoader:
    if batch_size < 1 or num_workers < 0:
        raise ValueError("Batch size must be positive and worker count nonnegative")
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, drop_last=False,
                      num_workers=num_workers, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed),
                      pin_memory=torch.device(device).type == "cuda")
