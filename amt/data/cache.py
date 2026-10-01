"""
Loading cached feature files (written by scripts/cache_maestro_features.py)
and sampling random training segments from them.

Layout: <cache_dir>/<split>/<stem>.npz with float16 `cqt` (88, T), uint8
`frame_roll` / `onset_roll` (88, T), and continuous reference notes
`ref_onsets` / `ref_offsets` / `ref_pitches`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Piece:
    stem: str
    cqt: np.ndarray        # float16 (88, T)
    frame: np.ndarray      # uint8   (88, T)
    onset: np.ndarray      # uint8   (88, T)
    ref_onsets: np.ndarray | None = None
    ref_offsets: np.ndarray | None = None
    ref_pitches: np.ndarray | None = None


def load_piece(path: Path, with_notes: bool = False) -> Piece:
    d = np.load(path)
    piece = Piece(
        stem=str(d["stem"]),
        cqt=d["cqt"],
        frame=d["frame_roll"].astype(np.uint8, copy=False),
        onset=d["onset_roll"].astype(np.uint8, copy=False),
    )
    if with_notes:
        piece.ref_onsets, piece.ref_offsets, piece.ref_pitches = (
            d["ref_onsets"], d["ref_offsets"], d["ref_pitches"],
        )
    return piece


def load_split(
    cache_dir: str | Path,
    split: str,
    max_hours: float | None = None,
    max_pieces: int | None = None,
    seed: int = 21,
    with_notes: bool = False,
    fs: float = 62.5,
) -> list[Piece]:
    """Load a split. If max_hours / max_pieces is given, take a seeded
    random subset (deterministic for a given seed)."""
    paths = sorted((Path(cache_dir) / split).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no .npz files in {Path(cache_dir) / split}")
    random.Random(seed).shuffle(paths)
    pieces: list[Piece] = []
    frames = 0
    for p in paths:
        if max_pieces is not None and len(pieces) >= max_pieces:
            break
        if max_hours is not None and frames / fs / 3600 >= max_hours:
            break
        piece = load_piece(p, with_notes=with_notes)
        pieces.append(piece)
        frames += piece.cqt.shape[1]
    return pieces


def total_hours(pieces: list[Piece], fs: float = 62.5) -> float:
    return sum(p.cqt.shape[1] for p in pieces) / fs / 3600


def compute_norm_stats(pieces: list[Piece]) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin mean/std of the log-CQT over all frames of `pieces`."""
    n_bins = pieces[0].cqt.shape[0]
    s = np.zeros(n_bins, dtype=np.float64)
    ss = np.zeros(n_bins, dtype=np.float64)
    n = 0
    for p in pieces:
        x = p.cqt.astype(np.float64)
        s += x.sum(axis=1)
        ss += (x * x).sum(axis=1)
        n += x.shape[1]
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean**2, 0.0))
    return mean.astype(np.float32), np.maximum(std, 1e-3).astype(np.float32)


class SegmentSampler:
    """Draws random fixed-length windows. Pieces are chosen in proportion to
    their length, so every moment of audio is equally likely to be seen."""

    def __init__(self, pieces: list[Piece], segment_frames: int, batch_size: int, seed: int = 21):
        self.pieces = [p for p in pieces if p.cqt.shape[1] > segment_frames]
        if not self.pieces:
            raise ValueError("no piece is longer than segment_frames")
        self.seg = segment_frames
        self.batch = batch_size
        n_starts = np.array([p.cqt.shape[1] - segment_frames + 1 for p in self.pieces], dtype=np.float64)
        self.probs = n_starts / n_starts.sum()
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.rng.choice(len(self.pieces), size=self.batch, p=self.probs)
        x = np.empty((self.batch, self.pieces[0].cqt.shape[0], self.seg), dtype=np.float32)
        yf = np.empty_like(x)
        yo = np.empty_like(x)
        for k, i in enumerate(idx):
            p = self.pieces[i]
            start = int(self.rng.integers(0, p.cqt.shape[1] - self.seg + 1))
            sl = slice(start, start + self.seg)
            x[k] = p.cqt[:, sl]
            yf[k] = p.frame[:, sl]
            yo[k] = p.onset[:, sl]
        return torch.from_numpy(x), torch.from_numpy(yf), torch.from_numpy(yo)
