"""
Tests for amt.baselines.
"""
from __future__ import annotations

import numpy as np

from amt.baselines import silence_prediction, spectral_peak_prediction


def test_silence_prediction_shape_and_all_zero():
    frame_roll, onset_roll = silence_prediction(n_pitches=88, n_frames=50)
    assert frame_roll.shape == (88, 50)
    assert onset_roll.shape == (88, 50)
    assert frame_roll.sum() == 0.0
    assert onset_roll.sum() == 0.0


def test_spectral_peak_activates_above_threshold_only():
    # min_duration_frames defaults to 5, so the active run needs to be
    # at least that long to survive the opening filter.
    cqt = np.full((3, 10), -80.0, dtype=np.float32)
    cqt[1, 2:7] = -10.0  # 5 consecutive loud frames
    frame_roll, _ = spectral_peak_prediction(cqt, threshold_db=-40.0)
    expected = np.zeros((3, 10), dtype=np.float32)
    expected[1, 2:7] = 1.0
    np.testing.assert_array_equal(frame_roll, expected)


def test_spectral_peak_onset_at_first_frame_if_active():
    cqt = np.full((2, 10), -10.0, dtype=np.float32)  # active from frame 0
    frame_roll, onset_roll = spectral_peak_prediction(cqt, threshold_db=-40.0)
    assert onset_roll[0, 0] == 1.0
    assert onset_roll[0, 1:].sum() == 0.0  # stays on, no repeated onset


def test_spectral_peak_onset_on_rising_edge_only():
    # Pitch 0: 3 off, 5 on, 2 off, 5 on -- two runs, each long enough to
    # survive opening; onsets expected at the start of each run.
    cqt = np.array(
        [[-80, -80, -80, -10, -10, -10, -10, -10, -80, -80, -10, -10, -10, -10, -10]],
        dtype=np.float32,
    )
    frame_roll, onset_roll = spectral_peak_prediction(cqt, threshold_db=-40.0)
    assert onset_roll[0, 3] == 1.0
    assert onset_roll[0, 10] == 1.0
    assert onset_roll[0].sum() == 2.0  # exactly these two onsets, nothing else


def test_spectral_peak_suppresses_short_flicker():
    # A single-frame blip, shorter than min_duration_frames, must be
    # suppressed -- this is the fix for the real-audio pathology where
    # a raw per-frame threshold produces hundreds of thousands of
    # spurious one-frame "notes" and makes note-level scoring hang.
    cqt = np.full((1, 10), -80.0, dtype=np.float32)
    cqt[0, 4] = -10.0  # one loud frame, isolated
    frame_roll, onset_roll = spectral_peak_prediction(cqt, threshold_db=-40.0, min_duration_frames=5)
    assert frame_roll.sum() == 0.0
    assert onset_roll.sum() == 0.0
