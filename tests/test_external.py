"""Tests for amt.evaluation.external (scoring another system's notes)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from amt.data.midi import Note, notes_to_piano_roll
from amt.evaluation.external import events_to_notes, score_external_notes


def _piece(notes, n_frames=200, fs=62.5):
    frame, _ = notes_to_piano_roll(notes, n_frames=n_frames, fs=fs, pitch_low=21, pitch_high=108)
    return SimpleNamespace(stem="p", frame=frame.astype(np.uint8),
                           ref_pitches=np.array([n.pitch for n in notes]),
                           ref_onsets=np.array([n.start for n in notes]),
                           ref_offsets=np.array([n.end for n in notes]))


def test_events_to_notes_reads_kong_format():
    ev = [{"onset_time": 1.0, "offset_time": 1.5, "midi_note": 60, "velocity": 70}]
    assert events_to_notes(ev) == [Note(60, 1.0, 1.5, 70)]


def test_perfect_external_notes_score_one():
    notes = [Note(60, 0.2, 0.8, 100), Note(64, 1.0, 2.0, 100)]
    r = score_external_notes(notes, _piece(notes))
    assert r["note"]["onset_f1"] == pytest.approx(1.0)
    assert r["note"]["onset_offset_f1"] == pytest.approx(1.0)
    assert r["frame"]["f1"] == pytest.approx(1.0)


def test_external_notes_are_scored_against_continuous_reference():
    ref = [Note(60, 0.20, 0.80, 100)]
    late = [Note(60, 0.30, 0.80, 100)]          # 100 ms late: outside the 50 ms tolerance
    assert score_external_notes(late, _piece(ref))["note"]["onset_f1"] == 0.0
    close = [Note(60, 0.24, 0.80, 100)]         # 40 ms: inside
    assert score_external_notes(close, _piece(ref))["note"]["onset_f1"] == pytest.approx(1.0)
