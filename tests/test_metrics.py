"""
Tests for amt.evaluation.metrics.

Covers the cases that actually matter for trusting these numbers in a
report: a perfect prediction scores 1.0, a completely wrong one scores
0.0 (not a crash or a NaN), and empty ref/est arrays -- silence, or a
model predicting nothing -- are handled explicitly rather than as
accidental edge cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from amt.data.midi import Note
from amt.evaluation.metrics import frame_metrics, note_metrics


# ---- frame_metrics ----------------------------------------------------

def test_frame_metrics_perfect_match():
    roll = np.zeros((10, 20), dtype=np.float32)
    roll[3, 5:10] = 1.0
    roll[7, 0:2] = 1.0
    m = frame_metrics(roll, roll.copy())
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_frame_metrics_completely_wrong():
    ref = np.zeros((10, 20), dtype=np.float32)
    ref[3, 5:10] = 1.0
    est = np.zeros((10, 20), dtype=np.float32)
    est[8, 0:5] = 1.0  # no overlap with ref at all
    m = frame_metrics(ref, est)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_frame_metrics_empty_ref_and_est_is_perfect():
    # Nothing to predict, nothing predicted -- a trivially correct case,
    # not an undefined 0/0.
    ref = np.zeros((10, 20), dtype=np.float32)
    est = np.zeros((10, 20), dtype=np.float32)
    m = frame_metrics(ref, est)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_frame_metrics_empty_estimate_has_zero_recall():
    ref = np.zeros((10, 20), dtype=np.float32)
    ref[3, 5:10] = 1.0
    est = np.zeros((10, 20), dtype=np.float32)  # predicts nothing
    m = frame_metrics(ref, est)
    assert m.recall == 0.0
    # precision is 0/0 here (nothing predicted) -> defined as 0.0
    assert m.precision == 0.0


def test_frame_metrics_partial_overlap():
    ref = np.zeros((1, 10), dtype=np.float32)
    ref[0, 0:6] = 1.0  # 6 active frames
    est = np.zeros((1, 10), dtype=np.float32)
    est[0, 3:9] = 1.0  # 6 active frames, overlapping ref in [3:6) -> 3 TP
    m = frame_metrics(ref, est)
    # TP=3, FP=3 (frames 6,7,8 predicted but not ref), FN=3 (frames 0,1,2 ref but not predicted)
    assert m.precision == pytest.approx(3 / 6)
    assert m.recall == pytest.approx(3 / 6)
    assert m.f1 == pytest.approx(0.5)


def test_frame_metrics_shape_mismatch_raises():
    ref = np.zeros((10, 20))
    est = np.zeros((10, 21))
    with pytest.raises(ValueError, match="shape"):
        frame_metrics(ref, est)


# ---- note_metrics -------------------------------------------------------

def test_note_metrics_perfect_match():
    notes = [
        Note(pitch=60, start=0.0, end=0.5, velocity=100),
        Note(pitch=64, start=0.6, end=1.0, velocity=100),
    ]
    m = note_metrics(notes, notes)
    assert m.onset_f1 == pytest.approx(1.0)
    assert m.onset_offset_f1 == pytest.approx(1.0)


def test_note_metrics_empty_estimate_scores_zero_recall():
    ref = [Note(pitch=60, start=0.0, end=0.5, velocity=100)]
    m = note_metrics(ref, [])
    assert m.onset_recall == 0.0
    assert m.onset_f1 == 0.0


def test_note_metrics_empty_reference_scores_zero_precision():
    est = [Note(pitch=60, start=0.0, end=0.5, velocity=100)]
    m = note_metrics([], est)
    assert m.onset_precision == 0.0
    assert m.onset_f1 == 0.0


def test_note_metrics_both_empty_is_defined_not_crashing():
    # mir_eval's own convention for "nothing to predict, nothing
    # predicted" -- just confirm this doesn't raise, and returns finite
    # numbers (mir_eval defines this as 0, not 1, unlike our frame
    # metric's convention -- documented here since it's easy to assume
    # the two metrics agree on this edge case when they don't).
    m = note_metrics([], [])
    assert np.isfinite(m.onset_f1)
    assert np.isfinite(m.onset_offset_f1)


def test_note_metrics_onset_only_ignores_wrong_offset():
    # Same pitch and onset, very different offset. Onset-only metric
    # should still count this as a match; onset+offset should not.
    ref = [Note(pitch=60, start=0.0, end=2.0, velocity=100)]
    est = [Note(pitch=60, start=0.01, end=0.3, velocity=100)]  # onset close, offset way early
    m = note_metrics(ref, est, onset_tolerance=0.05)
    assert m.onset_f1 == pytest.approx(1.0)
    assert m.onset_offset_f1 == 0.0


def test_note_metrics_onset_outside_tolerance_does_not_match():
    ref = [Note(pitch=60, start=0.0, end=0.5, velocity=100)]
    est = [Note(pitch=60, start=0.2, end=0.5, velocity=100)]  # 200ms off, tolerance is 50ms
    m = note_metrics(ref, est, onset_tolerance=0.05)
    assert m.onset_f1 == 0.0


def test_note_metrics_wrong_pitch_does_not_match():
    ref = [Note(pitch=60, start=0.0, end=0.5, velocity=100)]
    est = [Note(pitch=61, start=0.0, end=0.5, velocity=100)]  # one semitone off
    m = note_metrics(ref, est)
    assert m.onset_f1 == 0.0

def test_note_metrics_per_pitch_matches_global_mir_eval():
    import mir_eval
    from amt.evaluation.metrics import _notes_to_mir_eval_format

    rng = np.random.default_rng(0)
    for _ in range(5):
        ref = []
        for _ in range(150):
            s = float(rng.uniform(0, 20))
            ref.append(Note(int(rng.integers(40, 70)), s, s + float(rng.uniform(0.1, 1.5)), 100))
        est = []
        for n in ref:
            if rng.random() < 0.7:
                s = n.start + float(rng.normal(0, 0.04))
                est.append(Note(n.pitch, s, max(s + 0.05, n.end + float(rng.normal(0, 0.1))), 100))
        for _ in range(60):
            s = float(rng.uniform(0, 20))
            est.append(Note(int(rng.integers(40, 70)), s, s + 0.3, 100))

        ri, rp = _notes_to_mir_eval_format(ref)
        ei, ep = _notes_to_mir_eval_format(est)
        p0, r0, f0, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)
        p1, r1, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=0.2)

        m = note_metrics(ref, est)
        assert m.onset_precision == pytest.approx(p0)
        assert m.onset_recall == pytest.approx(r0)
        assert m.onset_f1 == pytest.approx(f0)
        assert m.onset_offset_precision == pytest.approx(p1)
        assert m.onset_offset_recall == pytest.approx(r1)
        assert m.onset_offset_f1 == pytest.approx(f1)