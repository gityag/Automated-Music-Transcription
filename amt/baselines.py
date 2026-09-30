"""
Baseline predictors, scored against the real model to show whether it's
actually learning anything beyond a trivial or hand-crafted heuristic.

Both baselines take a CQT array (or its shape) and return a
(frame_roll, onset_roll) pair in the same format the real model's
output will eventually be in, so they score through the exact same
amt.evaluation.metrics functions as any future model.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_opening


def silence_prediction(n_pitches: int, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Predicts no active notes at all, ever. The floor: any real system
    should beat this by a wide margin, and its recall is 0 by construction."""
    frame_roll = np.zeros((n_pitches, n_frames), dtype=np.float32)
    onset_roll = np.zeros((n_pitches, n_frames), dtype=np.float32)
    return frame_roll, onset_roll


def spectral_peak_prediction(
    cqt: np.ndarray,
    threshold_db: float = -40.0,
    min_duration_frames: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    A pitch is "active" in a frame if its log-magnitude CQT value in
    that frame exceeds threshold_db. This is the simplest possible
    audio-aware baseline: no learning, just "is there energy at this
    pitch's frequency right now" -- it will over-trigger on harmonics
    (a fundamental's energy leaks into bins at its overtones, which
    line up with the pitches above it on a log-frequency axis like the
    CQT's), so a good model should clearly beat it, especially on
    precision.

    cqt: log-magnitude CQT array, shape (n_pitches, n_frames), exactly
    as produced by amt.features.cqt.compute_log_cqt with cfg.cqt.fmin_midi
    == cfg.labels.pitch_low (so its rows already line up 1:1 with pitch
    numbers -- see ADR 0003). threshold_db is on the same dB scale
    compute_log_cqt uses (ref=1.0, i.e. absolute, not per-clip-relative).

    min_duration_frames: a raw per-frame threshold on real audio
    flickers on/off constantly (CQT magnitude isn't a clean square wave
    even during one sustained note), which without smoothing produces
    hundreds of thousands of spurious one-frame "notes" per song --
    slow to decode and slow (or effectively hung) to score, since
    mir_eval's note-matching cost grows badly with note count. A binary
    opening (erode-then-dilate) along the time axis removes any active
    run shorter than min_duration_frames before onsets are derived from
    it, which is the standard fix and also a more honest baseline: real
    piano notes don't last one 16ms frame.

    onset_roll is derived from the smoothed frame_roll: a 1 at any
    frame where a pitch transitions from inactive to active, which is
    what the decoder (amt.data.midi.piano_roll_to_notes) needs to tell
    separate re-triggered notes apart.
    """
    raw = cqt >= threshold_db
    structure = np.ones((1, min_duration_frames), dtype=bool)
    frame_roll = binary_opening(raw, structure=structure).astype(np.float32)

    onset_roll = np.zeros_like(frame_roll)
    onset_roll[:, 0] = frame_roll[:, 0]
    onset_roll[:, 1:] = np.clip(frame_roll[:, 1:] - frame_roll[:, :-1], 0, 1)

    return frame_roll, onset_roll
