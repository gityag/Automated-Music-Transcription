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
- Input normalization (per-bin mean/std from the training set) lives
  inside the model as buffers, so a checkpoint is self-contained and
  training/inference can't normalize differently.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class FrameCNN(nn.Module):
    def __init__(
        self,
        n_bins: int = 88,
        conv_channels=(32, 32, 64, 64),
        dropout: float = 0.25,
        onset_head: bool = False,
        hidden: int = 512,
        time_dilations=None,
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
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in * n_bins, hidden)
        self.frame_out = nn.Linear(hidden, n_bins)
        self.onset_out = nn.Linear(hidden, n_bins) if onset_head else None
        self.n_bins = n_bins
        self.receptive_radius = int(sum(dil))  # frames of context needed on each side

        self.register_buffer("norm_mean", torch.zeros(n_bins, 1))
        self.register_buffer("norm_std", torch.ones(n_bins, 1))

    def set_norm(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.norm_mean.copy_(torch.as_tensor(mean, dtype=torch.float32).reshape(-1, 1))
        self.norm_std.copy_(torch.as_tensor(std, dtype=torch.float32).reshape(-1, 1))

    def forward(self, x: torch.Tensor):
        x = (x - self.norm_mean) / self.norm_std            # (B, F, T)
        h = self.conv(x.unsqueeze(1))                        # (B, C, F, T)
        b, c, f, t = h.shape
        h = self.drop(h.permute(0, 3, 1, 2).reshape(b, t, c * f))
        h = self.drop(torch.relu(self.fc(h)))                # (B, T, hidden)
        frame = self.frame_out(h).transpose(1, 2)            # (B, 88, T)
        onset = self.onset_out(h).transpose(1, 2) if self.onset_out is not None else None
        return frame, onset


def build_model(cfg) -> FrameCNN:
    return FrameCNN(
        n_bins=cfg.cqt.n_bins,
        conv_channels=tuple(cfg.model.conv_channels),
        dropout=cfg.model.dropout,
        onset_head=cfg.model.onset_head,
    )


@torch.no_grad()
def predict_piece(model: FrameCNN, cqt: np.ndarray, chunk: int = 2048, device=None):
    """
    Run the model over a whole piece, in chunks (a full piece at once can
    exceed GPU memory). Each chunk is padded with `receptive_radius` extra
    frames of real context on each side and then cropped, so the result is
    identical to running the whole piece in one pass.

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
