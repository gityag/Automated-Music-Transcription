"""
Tests for amt.data.midi.
"""
from __future__ import annotations

import pretty_midi
import numpy as np
import pytest
from amt.data.midi import dedupe_notes

from amt.data.midi import (
    Note,
    load_notes,
    extend_notes_with_sustain_pedal,
    notes_to_piano_roll,
    piano_roll_to_notes,
)


def _make_midi(notes, pedal_events=None, path="/tmp/_test.mid"):
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
    path = _make_midi(
        notes=[(60, 0.0, 1.0, 100)],
        pedal_events=[(0.2, 100), (1.8, 0)],
    )
    extended = extend_notes_with_sustain_pedal(path)
    assert len(extended) == 1
    assert extended[0].end == pytest.approx(1.8)


def test_sustain_does_not_extend_past_pedal_up():
    path = _make_midi(
        notes=[(60, 0.0, 1.0, 100)],
        pedal_events=[(0.0, 100), (0.5, 0)],
    )
    extended = extend_notes_with_sustain_pedal(path)
    assert extended[0].end == pytest.approx(1.0)


def test_sustain_extension_stops_at_next_same_pitch_onset():
    path = _make_midi(
        notes=[(60, 0.0, 1.0, 100), (60, 1.2, 2.0, 100)],
        pedal_events=[(0.0, 100), (3.0, 0)],
    )
    extended = extend_notes_with_sustain_pedal(path)
    first = next(n for n in extended if n.start == 0.0)
    assert first.end == pytest.approx(1.2)


def test_repeated_notes_are_not_merged():
    path = _make_midi([(60, 0.0, 0.3, 100), (60, 0.3, 0.6, 100), (60, 0.6, 0.9, 100)])
    notes = load_notes(path)
    assert len(notes) == 3


def test_piano_roll_shape_and_onset_alignment():
    notes = [Note(pitch=21, start=0.0, end=0.5, velocity=100)]
    frame_roll, onset_roll = notes_to_piano_roll(
        notes, n_frames=100, fs=100, pitch_low=21, pitch_high=108
    )
    assert frame_roll.shape == (88, 100)
    assert onset_roll.shape == (88, 100)
    assert onset_roll[0, 0] == 1.0
    assert onset_roll[0, 1:].sum() == 0.0
    assert frame_roll[0, 0:50].sum() == 50.0
    assert frame_roll[0, 50:].sum() == 0.0


def test_piano_roll_no_false_gap_on_held_note():
    notes = [Note(pitch=60, start=0.0, end=1.0, velocity=100)]
    frame_roll, _ = notes_to_piano_roll(
        notes, n_frames=100, fs=100, pitch_low=21, pitch_high=108
    )
    row = 60 - 21
    assert np.all(frame_roll[row, :] == 1.0)


def test_piano_roll_ignores_out_of_range_pitch():
    notes = [Note(pitch=10, start=0.0, end=0.5, velocity=100)]
    frame_roll, onset_roll = notes_to_piano_roll(
        notes, n_frames=100, fs=100, pitch_low=21, pitch_high=108
    )
    assert frame_roll.sum() == 0.0
    assert onset_roll.sum() == 0.0


def test_piano_roll_clips_note_starting_at_or_past_end():
    notes = [Note(pitch=60, start=2, end=2.1, velocity=100)]
    frame_roll, onset_roll = notes_to_piano_roll(
        notes, n_frames=100, fs=100, pitch_low=21, pitch_high=108
    )
    row = 60 - 21
    assert onset_roll[row, 99] == 1.0


# ---- piano_roll_to_notes (decoder) --------------------------------------

def test_decode_round_trip_single_note():
    original = [Note(pitch=60, start=0.1, end=0.5, velocity=100)]
    frame_roll, onset_roll = notes_to_piano_roll(
        original, n_frames=100, fs=100, pitch_low=21, pitch_high=108
    )
    decoded = piano_roll_to_notes(frame_roll, onset_roll, fs=100, pitch_low=21)
    assert len(decoded) == 1
    assert decoded[0].pitch == 60
    # Frame-quantized, so recovered times land on the frame grid near
    # the originals, not necessarily bit-identical.
    assert decoded[0].start == pytest.approx(0.1, abs=0.01)
    assert decoded[0].end == pytest.approx(0.5, abs=0.01)


def test_decode_round_trip_chord():
    original = [
        Note(pitch=60, start=0.0, end=0.5, velocity=100),
        Note(pitch=64, start=0.0, end=0.5, velocity=100),
        Note(pitch=67, start=0.0, end=0.3, velocity=100),
    ]
    frame_roll, onset_roll = notes_to_piano_roll(
        original, n_frames=100, fs=100, pitch_low=21, pitch_high=108
    )
    decoded = piano_roll_to_notes(frame_roll, onset_roll, fs=100, pitch_low=21)
    assert sorted(n.pitch for n in decoded) == [60, 64, 67]


def test_decode_separates_repeated_notes_with_no_gap():
    # Two consecutive notes on the same pitch, second starting exactly
    # where the first ends -- frame_roll alone would look like one
    # continuous note; onset_roll's second onset must split them.
    original = [
        Note(pitch=60, start=0.0, end=0.5, velocity=100),
        Note(pitch=60, start=0.5, end=1.0, velocity=100),
    ]
    frame_roll, onset_roll = notes_to_piano_roll(
        original, n_frames=100, fs=100, pitch_low=21, pitch_high=108
    )
    decoded = piano_roll_to_notes(frame_roll, onset_roll, fs=100, pitch_low=21)
    assert len(decoded) == 2
    assert decoded[0].end == pytest.approx(decoded[1].start, abs=0.01)


def test_decode_empty_roll_gives_no_notes():
    frame_roll = np.zeros((88, 100), dtype=np.float32)
    onset_roll = np.zeros((88, 100), dtype=np.float32)
    decoded = piano_roll_to_notes(frame_roll, onset_roll, fs=100, pitch_low=21)
    assert decoded == []

def test_dedupe_removes_exact_duplicates_keeps_longest():
    notes = [
        Note(60, 1.0, 1.5, 100),
        Note(60, 1.0, 2.0, 100),
        Note(64, 1.0, 1.5, 100),
    ]
    out = dedupe_notes(notes)
    assert len(out) == 2
    assert next(n for n in out if n.pitch == 60).end == 2.0


def test_dedupe_keeps_genuine_restrikes():
    notes = [Note(60, 0.0, 0.2, 100), Note(60, 0.25, 0.45, 100), Note(60, 0.5, 0.7, 100)]
    assert len(dedupe_notes(notes)) == 3