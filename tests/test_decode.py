"""Tests for amt.evaluation.decode (pure NumPy)."""
from __future__ import annotations

import numpy as np
import pytest

from amt.data.midi import Note, notes_to_piano_roll
from amt.evaluation.decode import decode_notes, score_predictions

FS = 100.0


def _roll(notes, n_frames=200):
    return notes_to_piano_roll(notes, n_frames=n_frames, fs=FS, pitch_low=21, pitch_high=108)


def test_frame_only_round_trip():
    notes = [Note(60, 0.2, 0.8, 100), Note(64, 0.5, 1.2, 100)]
    frame, _ = _roll(notes)
    out = decode_notes(frame, None, fs=FS, pitch_low=21)
    assert [(n.pitch, round(n.start, 2), round(n.end, 2)) for n in out] == [(60, 0.2, 0.8), (64, 0.5, 1.2)]


def test_min_frames_drops_flicker():
    frame = np.zeros((88, 100), dtype=np.float32)
    frame[10, 20:22] = 1.0   # 2 frames: too short
    frame[11, 20:30] = 1.0   # 10 frames: kept
    out = decode_notes(frame, None, fs=FS, pitch_low=21, min_frames=3)
    assert [n.pitch for n in out] == [21 + 11]


def test_onset_head_splits_repeated_note_while_frame_stays_on():
    frame = np.zeros((88, 100), dtype=np.float32)
    onset = np.zeros((88, 100), dtype=np.float32)
    frame[5, 10:60] = 1.0     # one long continuous frame activation...
    onset[5, 10] = 0.9        # ...with two onsets inside it
    onset[5, 35] = 0.9
    out = decode_notes(frame, onset, fs=FS, pitch_low=21)
    assert len(out) == 2
    assert out[0].end == pytest.approx(out[1].start)
    assert decode_notes(frame, None, fs=FS, pitch_low=21).__len__() == 1  # frame-only can't split


def test_onset_head_ignores_frames_without_onset():
    frame = np.zeros((88, 100), dtype=np.float32)
    frame[5, 10:60] = 1.0
    onset = np.zeros((88, 100), dtype=np.float32)
    assert decode_notes(frame, onset, fs=FS, pitch_low=21) == []


def test_onset_peak_picking_gives_one_onset_per_peak():
    frame = np.ones((88, 100), dtype=np.float32) * 0
    frame[5, 10:40] = 1.0
    onset = np.zeros((88, 100), dtype=np.float32)
    onset[5, 9:13] = [0.6, 0.9, 0.7, 0.6]   # a smeared onset: should count once
    out = decode_notes(frame, onset, fs=FS, pitch_low=21)
    assert len(out) == 1 and out[0].start == pytest.approx(0.10)


class _P:
    def __init__(self, notes, n_frames=200):
        self.frame, self.onset = _roll(notes, n_frames)
        self.ref_pitches = np.array([n.pitch for n in notes])
        self.ref_onsets = np.array([n.start for n in notes])
        self.ref_offsets = np.array([n.end for n in notes])


def test_perfect_predictions_score_one():
    notes = [Note(60, 0.2, 0.8, 100), Note(64, 0.5, 1.2, 100), Note(60, 1.3, 1.7, 100)]
    p = _P(notes)
    res = score_predictions([p], [(p.frame.copy(), p.onset.copy())], fs=FS, pitch_low=21)
    assert res["frame"]["f1"] == pytest.approx(1.0)
    assert res["note"]["onset_f1"] == pytest.approx(1.0)
    assert res["note"]["onset_offset_f1"] == pytest.approx(1.0)