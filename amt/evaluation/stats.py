"""
Uncertainty for per-piece averaged scores (pure NumPy).

The test sets are small (177 MAESTRO pieces, 50 SMD pieces), so a mean
F1 on its own hides how much it would move with a different sample of
pieces. These helpers resample PIECES (not notes or frames: notes within
a piece are not independent) with a fixed seed, so every reported
interval is reproducible.

- bootstrap_ci: 95% interval for one system's mean score.
- paired_bootstrap: interval for the mean difference between two systems
  scored on the SAME pieces, plus the share of resamples in which the
  difference is <= 0. Pairing removes the piece-difficulty variance both
  systems share, so it detects real differences a comparison of two
  separate intervals would miss.
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("values must be a non-empty 1-D sequence")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def paired_bootstrap(a, b, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Mean of (b - a) over pieces, its interval, and p_le_zero = fraction of
    resamples with mean difference <= 0 (small = b reliably better)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("a and b must be non-empty 1-D sequences of equal length")
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_resamples, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "p_le_zero": float(np.mean(means <= 0)), "n": int(len(d))}


def summarize(per_piece: list[dict], seed: int = 0) -> dict:
    """95% intervals for the headline metrics from score_predictions(..., per_piece=True)."""
    keys = {"frame_f1": ("frame", "f1"),
            "note_onset_f1": ("note", "onset_f1"),
            "note_onset_offset_f1": ("note", "onset_offset_f1")}
    return {name: bootstrap_ci([p[g][k] for p in per_piece], seed=seed) for name, (g, k) in keys.items()}
