"""
Frame-level CNN for piano transcription from a log-CQT.

Input  (B, 88, T): log-magnitude CQT, one bin per semitone (A0..C8).
Output (B, 88, T): frame logits (+ onset logits if onset_head=True).

Design, and why:
- Conv stack over (frequency, time) with 3x3 kernels. Frequency is NOT
  pooled: the output must stay aligned 1:1 with the 88 CQT bins/pitches.
  Time dilations 1,2,4,8 widen the temporal receptive field to +-15
  frames (~+-240 ms) without pooling, so the output keeps the full
  62.5 fps frame rate.
- 3x3 convs only see +-4 semitones, but a piano note's harmonics sit at
  +12, +19, +24 ... bins. So after the convs, every frame's whole
  (channels x 88 bins) column goes through a fully-connected layer: each
  pitch's decision can use all harmonics (the Onsets-and-Frames layout).
- Optional temporal model (temporal="bigru"): the per-frame hidden vector
  is projected to `pitch_feat` features per pitch, and ONE small
  bidirectional GRU, shared by all 88 pitches, runs over time on each
  pitch's feature sequence (the frequency-grouped recurrence of HPPNet,
  Wei et al. 2022). It gives every pitch a memory of how long it has
  been sounding, which is what offsets need; the CNN alone sees only
  +-240 ms. It is RESIDUAL: the CNN's own frame/onset logits (the
  temporal="none" heads) are also fed to the GRU, and the GRU's output
  is ADDED to them through zero-initialised heads. At step 0 the model
  is exactly the CNN model, and the GRU only learns corrections. (Without
  this, the GRU path learned far slower than the CNN alone.)
- Optional harmonic block (harmonic=True): a harmonic dilated convolution
  (HarmonicDilatedConv) added residually after the first conv layer, so
  early features combine each bin with its harmonics' bins.
- Input normalization (per-bin mean/std from the training set) lives
  inside the model as buffers, so a checkpoint is self-contained and
  training/inference can't normalize differently.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def harmonic_dilations(bins_per_octave: int = 12, n_harmonics: int = 8) -> list[int]:
    """Bin offsets of harmonics 2..n above a fundamental on a log-frequency
    axis: round(bins_per_octave * log2(k)). 12 bins/octave -> 12, 19, 24, 28,
    31, 34, 36."""
    return [round(bins_per_octave * math.log2(k)) for k in range(2, n_harmonics + 1)]


class HarmonicDilatedConv(nn.Module):
    """Harmonic dilated convolution (HPPNet, Wei et al. 2022), on our CQT.

    Output bin f is a learned sum over input bins f, f+d_2, ..., f+d_8 (its
    harmonics), each with a 3-frame time kernel. A 3x3 conv only sees +-1
    semitone per layer; this lets the very first layers combine a note's
    fundamental with its overtones. Bins above the top of the range are
    zero-padded."""

    def __init__(self, c_in: int, c_out: int, dilations: list[int]):
        super().__init__()
        self.dilations = list(dilations)
        self.base = nn.Conv2d(c_in, c_out, (1, 3), padding=(0, 1))
        self.harm = nn.ModuleList(
            nn.Conv2d(c_in, c_out, (1, 3), padding=(0, 1), bias=False) for _ in self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, F, T)
        out = self.base(x)
        n_bins = x.shape[2]
        for d, conv in zip(self.dilations, self.harm):
            shifted = F.pad(x[:, :, d:, :], (0, 0, 0, min(d, n_bins)))  # bin f now holds bin f+d
            out = out + conv(shifted)
        return out


class FrameCNN(nn.Module):
    def __init__(
        self,
        n_bins: int = 88,
        conv_channels=(32, 32, 64, 64),
        dropout: float = 0.25,
        onset_head: bool = False,
        hidden: int = 512,
        time_dilations=None,
        temporal: str = "none",
        pitch_feat: int = 16,
        rnn_hidden: int = 32,
        harmonic: bool = False,
        bins_per_octave: int = 12,
    ):
        super().__init__()
        dil = list(time_dilations) if time_dilations else [2**i for i in range(len(conv_channels))]
        layers: list[nn.Module] = []
        c_in = 1
        for c_out, d in zip(conv_channels, dil):
            layers += [
                nn.Conv2d(c_in, c_out, (3, 3), padding=(1, d), dilation=(1, d)),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        # optional harmonic block, applied residually after the first conv layer
        self.harmonic = None
        if harmonic:
            c1 = conv_channels[0]
            self.harmonic = nn.Sequential(
                HarmonicDilatedConv(c1, c1, harmonic_dilations(bins_per_octave)),
                nn.BatchNorm2d(c1), nn.ReLU(inplace=True))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in * n_bins, hidden)
        if temporal not in ("none", "bigru"):
            raise ValueError(f"unknown temporal model {temporal!r}")
        self.temporal = temporal
        self.n_bins = n_bins
        self.frame_out = nn.Linear(hidden, n_bins)
        self.onset_out = nn.Linear(hidden, n_bins) if onset_head else None
        self.receptive_radius = int(sum(dil))  # frames of context needed on each side
        if temporal == "bigru":
            self.pitch_feat = pitch_feat
            self.pitch_proj = nn.Linear(hidden, n_bins * pitch_feat)
            n_in = pitch_feat + (2 if onset_head else 1)   # + the CNN's own logits
            self.rnn = nn.GRU(n_in, rnn_hidden, batch_first=True, bidirectional=True)
            self.frame_rnn_out = nn.Linear(2 * rnn_hidden, 1)
            self.onset_rnn_out = nn.Linear(2 * rnn_hidden, 1) if onset_head else None
            for head in (self.frame_rnn_out, self.onset_rnn_out):
                if head is not None:
                    nn.init.zeros_(head.weight)
                    nn.init.zeros_(head.bias)
            # A GRU's context is unbounded; predict_piece overlaps chunks by
            # this many frames (~8 s), far longer than its effective memory.
            self.receptive_radius = 512

        self.register_buffer("norm_mean", torch.zeros(n_bins, 1))
        self.register_buffer("norm_std", torch.ones(n_bins, 1))

    def set_norm(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.norm_mean.copy_(torch.as_tensor(mean, dtype=torch.float32).reshape(-1, 1))
        self.norm_std.copy_(torch.as_tensor(std, dtype=torch.float32).reshape(-1, 1))

    def forward(self, x: torch.Tensor):
        x = (x - self.norm_mean) / self.norm_std            # (B, F, T)
        if self.harmonic is None:
            h = self.conv(x.unsqueeze(1))                    # (B, C, F, T)
        else:
            h = self.conv[:3](x.unsqueeze(1))                # first conv + BN + ReLU
            h = h + self.harmonic(h)
            h = self.conv[3:](h)
        b, c, f, t = h.shape
        h = self.drop(h.permute(0, 3, 1, 2).reshape(b, t, c * f))
        h = self.drop(torch.relu(self.fc(h)))                # (B, T, hidden)
        frame = self.frame_out(h)                            # (B, T, 88)
        onset = self.onset_out(h) if self.onset_out is not None else None
        if self.temporal == "bigru":
            z = torch.relu(self.pitch_proj(h)).view(b, t, self.n_bins, self.pitch_feat)
            extra = [frame.unsqueeze(-1)] + ([onset.unsqueeze(-1)] if onset is not None else [])
            z = torch.cat([z] + extra, dim=-1)               # (B, T, 88, n_in)
            z = z.permute(0, 2, 1, 3).reshape(b * self.n_bins, t, z.shape[-1])
            y, _ = self.rnn(z)                               # (B*88, T, 2*rnn_hidden)
            y = self.drop(y)
            frame = frame + self.frame_rnn_out(y).view(b, self.n_bins, t).transpose(1, 2)
            if onset is not None:
                onset = onset + self.onset_rnn_out(y).view(b, self.n_bins, t).transpose(1, 2)
        frame = frame.transpose(1, 2)                        # (B, 88, T)
        onset = onset.transpose(1, 2) if onset is not None else None
        return frame, onset


def build_model(cfg) -> FrameCNN:
    return FrameCNN(
        n_bins=cfg.cqt.n_bins,
        conv_channels=tuple(cfg.model.conv_channels),
        dropout=cfg.model.dropout,
        onset_head=cfg.model.onset_head,
        temporal=getattr(cfg.model, "temporal", "none"),
        pitch_feat=getattr(cfg.model, "pitch_feat", 16),
        rnn_hidden=getattr(cfg.model, "rnn_hidden", 32),
        harmonic=getattr(cfg.model, "harmonic", False),
        bins_per_octave=cfg.cqt.bins_per_octave,
    )


@torch.no_grad()
def predict_piece(model: FrameCNN, cqt: np.ndarray, chunk: int = 2048, device=None):
    """
    Run the model over a whole piece, in chunks (a full piece at once can
    exceed GPU memory). Each chunk is padded with `receptive_radius` extra
    frames of real context on each side and then cropped, so the result is
    identical to running the whole piece in one pass (for the GRU model:
    equal within float tolerance, since 512 frames of overlap far exceed
    its effective memory).

    Returns (frame_probs, onset_probs_or_None), each float32 (88, T).
    """
    model.eval()
    device = device or next(model.parameters()).device
    ctx = model.receptive_radius
    n_bins, n_frames = cqt.shape
    frame = np.zeros((n_bins, n_frames), dtype=np.float32)
    onset = np.zeros((n_bins, n_frames), dtype=np.float32) if model.onset_out is not None else None

    for s in range(0, n_frames, chunk):
        e = min(n_frames, s + chunk)
        a, b = max(0, s - ctx), min(n_frames, e + ctx)
        x = torch.from_numpy(np.asarray(cqt[:, a:b], dtype=np.float32))[None].to(device)
        f, o = model(x)
        frame[:, s:e] = torch.sigmoid(f)[0, :, s - a : e - a].float().cpu().numpy()
        if onset is not None:
            onset[:, s:e] = torch.sigmoid(o)[0, :, s - a : e - a].float().cpu().numpy()
    return frame, onset
