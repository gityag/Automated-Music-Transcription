"""Seed every RNG a training or preprocessing run touches, from one call."""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # MPS (Apple Silicon) has no separate manual_seed call as of
        # torch 2.1; torch.manual_seed above covers it.
    except ImportError:
        pass  # torch not installed yet (e.g. during preprocessing-only steps)
