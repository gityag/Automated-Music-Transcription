"""
Tests for amt.data.midi.

Several of these encode bugs found in the legacy pipeline as regression
tests: held notes getting a false gap at segment boundaries, re-struck
notes being merged, and sustain-pedal extension being applied
inconsistently. A synthetic pretty_midi.PrettyMIDI object is built in
memory for each case rather than relying on a fixture file, so the test
doubles as documentation of exactly what's being checked.
"""
from __future__ import annotations

import pretty_midi
import numpy as np
import pytest

from amt.data.midi import (
    Note,
    load_notes,
    extend_notes_with_sustain_pedal,
    notes_to_piano_roll,
)


def _make_midi(notes, pedal_events=None, path="/tmp/_test.mid"):
    """notes: list of (pitch, start, end, velocity).
    pedal_events: list of (time, value) CC64 events.
    """
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end, velocity in notes:
        inst.notes.append(
            pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)
        )
    for time, value in pedal_events or []:
        inst.control_changes.append(
            pretty_midi.ControlChange(number=64, value=value, time=time)
        )
    pm.instruments.append(inst)
    pm.write(path)
    return path


def test_load_notes_basic():
    path = _make_midi([(60, 0.0, 1.0, 100), (64, 0.5, 1.5, 90)])
    notes = load_notes(path)
    assert notes == [
        Note(pitch=60, start=0.0, end=1.0, velocity=100),
        Note(pitch=64, start=0.5, end=1.5, velocity=90),
    ]


def test_sustain_extends_note_held_at_pedal_down():
    # Note ends at 1.0s while the pedal (down at 0.2s) is still held,
    # and released at 1.8s -- the note should ring until the release.
    path = _make_midi(
        notes=[(60, 0.0, 1.0, 100)],
        pedal_events=[(0.2, 100), (1.8, 0)],
    )
    extended = extend_notes_with_sustain_pedal(path)
    assert len(extended) == 1
    assert extended[0].end == pytest.approx(1.8)


def test_sustain_does_not_extend_past_pedal_up():
    # Note ends at 1.0s, pedal already released at 0.5s -> no extension.
    path = _make_midi(
        notes=[(60, 0.0, 1.0, 100)],
        pedal_events=[(0.0, 100), (0.5, 0)],
    )
    extended = extend_notes_with_sustain_pedal(path)
    assert extended[0].end == pytest.approx(1.0)


def test_sustain_extension_stops_at_next_same_pitch_onset():
    # Same key struck again at 1.2s while the pedal is still down from
    # the first note -- the first note's ring must not swallow the
    # second onset just because the pedal never lifted.
    path = _make_midi(
        notes=[(60, 0.0, 1.0, 100), (60, 1.2, 2.0, 100)],
        pedal_events=[(0.0, 100), (3.0, 0)],
    )
    extended = extend_notes_with_sustain_pedal(path)
    first = next(n for n in extended if n.start == 0.0)
    assert first.end == pytest.approx(1.2)


def test_repeated_notes_are_not_merged():
    # The legacy `chop_simplified_midi` dropped a note_on for a pitch
    # that was already "on", merging fast repeated notes into one.
    # pretty_midi keeps each note event distinct regardless of overlap.
    path = _make_midi([(60, 0.0, 0.3, 100), (60, 0.3, 0.6, 100), (60, 0.6, 0.9, 100)])
    notes = load_notes(path)
    assert len(notes) == 3


def test_piano_roll_shape_and_onset_alignment():
    notes = [Note(pitch=21, start=0.0, end=0.5, velocity=100)]
    frame_roll, onset_roll = notes_to_piano_roll(
        notes, duration=1.0, fs=100, pitch_low=21, pitch_high=108
    )
    assert frame_roll.shape == (88, 100)
    assert onset_roll.shape == (88, 100)
    assert onset_roll[0, 0] == 1.0
    assert onset_roll[0, 1:].sum() == 0.0  # exactly one onset frame
    assert frame_roll[0, 0:50].sum() == 50.0  # active for the full 0.5s
    assert frame_roll[0, 50:].sum() == 0.0


def test_piano_roll_no_false_gap_on_held_note():
    # Regression test for the legacy encoder, which inserted a spurious
    # note-off near a segment boundary for notes meant to keep sounding.
    # A note spanning the whole duration should have zero gaps.
    notes = [Note(pitch=60, start=0.0, end=1.0, velocity=100)]
    frame_roll, _ = notes_to_piano_roll(
        notes, duration=1.0, fs=100, pitch_low=21, pitch_high=108
    )
    row = 60 - 21
    assert np.all(frame_roll[row, :] == 1.0)


def test_piano_roll_ignores_out_of_range_pitch():
    notes = [Note(pitch=10, start=0.0, end=0.5, velocity=100)]  # below A0
    frame_roll, onset_roll = notes_to_piano_roll(
        notes, duration=1.0, fs=100, pitch_low=21, pitch_high=108
    )
    assert frame_roll.sum() == 0.0
    assert onset_roll.sum() == 0.0
