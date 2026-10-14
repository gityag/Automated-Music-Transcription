"""
Controlled audio degradations for robustness testing (NumPy/SciPy only,
seeded, so every condition is reproducible).

Each function takes mono audio at `sr` and returns (audio, time_scale):
time_scale multiplies every reference time (1.0 unless the degradation
changes the speed). Detuning is done by resampling, which shifts pitch AND
speed together, the way a slightly mis-tuned or mis-clocked recording
behaves; reference times are rescaled so the labels stay aligned, while
reference PITCHES stay the same (the piano is simply out of tune).
"""
from __future__ import annotations

import librosa
import numpy as np
from scipy import signal


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def add_noise(x: np.ndarray, sr: int, snr_db: float, seed: int = 0):
    """White Gaussian noise at the given signal-to-noise ratio (whole piece)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(x))
    noise *= _rms(x) / _rms(noise) / (10 ** (snr_db / 20))
    return (x + noise).astype(np.float32), 1.0


def add_reverb(x: np.ndarray, sr: int, rt60: float, seed: int = 0):
    """Synthetic room: direct sound plus an exponentially decaying noise tail
    (60 dB decay in rt60 seconds). Output keeps the input's length and RMS."""
    rng = np.random.default_rng(seed)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.908 * t / rt60) * 0.3
    ir[0] = 1.0
    y = signal.fftconvolve(x, ir)[: len(x)]
    return (y * _rms(x) / _rms(y)).astype(np.float32), 1.0


def lowpass(x: np.ndarray, sr: int, cutoff_hz: float):
    """8th-order Butterworth low-pass (zero phase): a dull, phone-like recording."""
    sos = signal.butter(8, cutoff_hz, btype="low", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32), 1.0


def detune(x: np.ndarray, sr: int, cents: float):
    """Raise pitch by `cents` (lower if negative) via resampling. Playback is
    faster by r = 2**(cents/1200), so reference times scale by 1/r."""
    r = 2 ** (cents / 1200)
    y = librosa.resample(np.asarray(x, dtype=np.float32), orig_sr=sr * r, target_sr=sr, res_type="soxr_hq")
    return y.astype(np.float32), 1.0 / r


CONDITIONS = {
    "clean": lambda x, sr: (x.astype(np.float32), 1.0),
    "noise_20db": lambda x, sr: add_noise(x, sr, 20),
    "noise_10db": lambda x, sr: add_noise(x, sr, 10),
    "reverb_0.5s": lambda x, sr: add_reverb(x, sr, 0.5),
    "reverb_1.5s": lambda x, sr: add_reverb(x, sr, 1.5),
    "lowpass_4khz": lambda x, sr: lowpass(x, sr, 4000),
    "detune_+25c": lambda x, sr: detune(x, sr, 25),
    "detune_-25c": lambda x, sr: detune(x, sr, -25),
    "detune_+50c": lambda x, sr: detune(x, sr, 50),
}
