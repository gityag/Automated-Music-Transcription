"""
Tests for amt.features.cqt.

The phase-invariance test directly targets the bug this module fixes:
the legacy pipeline kept only the real part of the complex CQT, which
changes with the phase of the input signal even when the signal's pitch
content is identical.
"""
from __future__ import annotations

import numpy as np

from amt.features.cqt import compute_log_cqt, frame_rate

SR = 16_000
FMIN_MIDI = 21
N_BINS = 88
BINS_PER_OCTAVE = 12
HOP_LENGTH = 256


def _sine(freq_hz: float, duration_s: float, sr: int = SR, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return 0.5 * np.sin(2 * np.pi * freq_hz * t + phase).astype(np.float32)


def _cqt(audio: np.ndarray) -> np.ndarray:
    return compute_log_cqt(audio, SR, FMIN_MIDI, N_BINS, BINS_PER_OCTAVE, HOP_LENGTH)


def test_shape_and_dtype():
    audio = _sine(440.0, duration_s=1.0)
    out = _cqt(audio)
    assert out.dtype == np.float32
    assert out.shape[0] == N_BINS


def test_deterministic():
    audio = _sine(440.0, duration_s=0.5)
    a = _cqt(audio)
    b = _cqt(audio)
    np.testing.assert_array_equal(a, b)


def test_silence_is_not_nan_or_inf():
    audio = np.zeros(SR, dtype=np.float32)
    out = _cqt(audio)
    assert np.all(np.isfinite(out))


def test_phase_invariance():
    # This is the regression test for the real-part-only bug: two
    # sine waves at the same frequency and amplitude, differing only
    # in starting phase, must produce (nearly) the same log-magnitude
    # CQT. The legacy `.real` feature would not pass this.
    a = _cqt(_sine(440.0, duration_s=1.0, phase=0.0))
    b = _cqt(_sine(440.0, duration_s=1.0, phase=np.pi / 2))
    # Restrict the comparison to bins that actually carry the note's
    # energy (near its fundamental, MIDI 69 -> row 48) and to interior
    # frames. Bins far from the fundamental hold only numerical noise
    # floor, which is expected to be phase-sensitive since it isn't
    # signal -- comparing it would make this a noise-floor test, not a
    # phase-invariance test.
    signal_rows = slice(44, 53)
    interior_frames = slice(4, -4)
    np.testing.assert_allclose(
        a[signal_rows, interior_frames], b[signal_rows, interior_frames], atol=2.0
    )


def test_frame_rate_matches_hop_length():
    assert frame_rate(SR, HOP_LENGTH) == SR / HOP_LENGTH
