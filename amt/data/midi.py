"""
MIDI parsing and label construction.

Replaces the legacy `create_simplified_midi` / `chop_simplified_midi` /
`encode_midi_segment` / `decode_midi` chain. That chain hand-rolled tempo
and tick math (breaking on multi-tempo or multi-track files), deleted
re-struck notes at the same pitch, and produced boundary artifacts on
held notes (see docs/decisions/0001-pretty-midi-labels.md for the
specific bugs and why this design replaces them).

pretty_midi handles tempo maps, `note_on velocity=0` as note-off, and
tick-to-second conversion internally, so this module only has to encode
the *musical* decision this project makes: how sustain pedal extends a
note's sounding duration. That rule is applied identically whether the
notes come from ground truth or from decoding a model's predictions, so
train-time and eval-time labels are never allowed to diverge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pretty_midi


@dataclass(frozen=True)
class Note:
    pitch: int          # MIDI note number, e.g. 60 = middle C
    start: float         # seconds
    end: float           # seconds
    velocity: int


def load_notes(midi_path: str) -> list[Note]:
    """Load note events from a MIDI file, in absolute seconds.

    pretty_midi already merges `note_on velocity=0` into note-off and
    resolves the file's tempo map, so this is just a flat extraction.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = [
        Note(pitch=n.pitch, start=n.start, end=n.end, velocity=n.velocity)
        for instrument in pm.instruments
        for n in instrument.notes
    ]
    return sorted(notes, key=lambda n: (n.start, n.pitch))


def extend_notes_with_sustain_pedal(
    midi_path: str,
    threshold: int = 64,
) -> list[Note]:
    """Extend each note's `end` while the sustain pedal (CC64) is held.

    Standard rule (matching Hawthorne et al., Onsets and Frames): if a
    note ends while the pedal is down, its sounding end is pushed out to
    the pedal release -- but never past the next note-on of the *same
    pitch*, since re-striking a key ends the previous ring regardless of
    the pedal. Applying the wrong rule, or applying it only sometimes, is
    a common and easy-to-miss source of note-offset evaluation error.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    all_notes: list[Note] = []

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        pedal_events = sorted(
            (cc.time, cc.value)
            for cc in instrument.control_changes
            if cc.number == 64
        )
        pedal_down_intervals = _pedal_down_intervals(pedal_events, threshold)

        notes = sorted(instrument.notes, key=lambda n: n.start)
        next_onset_by_pitch: dict[int, list[float]] = {}
        for n in notes:
            next_onset_by_pitch.setdefault(n.pitch, []).append(n.start)

        for n in notes:
            end = n.end
            interval = _interval_containing(pedal_down_intervals, end)
            if interval is not None:
                pedal_release = interval[1]
                same_pitch_onsets = [
                    t for t in next_onset_by_pitch[n.pitch] if t > n.start
                ]
                next_same_pitch = min(same_pitch_onsets, default=float("inf"))
                end = min(pedal_release, next_same_pitch)
            all_notes.append(
                Note(pitch=n.pitch, start=n.start, end=end, velocity=n.velocity)
            )

    return sorted(all_notes, key=lambda n: (n.start, n.pitch))


def _pedal_down_intervals(
    pedal_events: list[tuple[float, int]], threshold: int
) -> list[tuple[float, float]]:
    """Collapse a stream of CC64 events into (down_time, up_time) intervals."""
    intervals: list[tuple[float, float]] = []
    down_since: float | None = None
    for time, value in pedal_events:
        is_down = value >= threshold
        if is_down and down_since is None:
            down_since = time
        elif not is_down and down_since is not None:
            intervals.append((down_since, time))
            down_since = None
    if down_since is not None:
        intervals.append((down_since, float("inf")))
    return intervals


def _interval_containing(
    intervals: list[tuple[float, float]], t: float
) -> tuple[float, float] | None:
    for start, end in intervals:
        if start <= t <= end:
            return (start, end)
    return None


def notes_to_piano_roll(
    notes: list[Note],
    duration: float,
    fs: int,
    pitch_low: int,
    pitch_high: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize notes to frame-aligned frame-activity and onset rolls.

    Returns (frame_roll, onset_roll), each shape (n_pitches, n_frames),
    dtype float32, values in {0, 1}. `fs` is frames per second and must
    match the CQT hop rate the audio features use (see CQTConfig), or
    input and target end up on different time grids.
    """
    n_pitches = pitch_high - pitch_low + 1
    n_frames = int(np.ceil(duration * fs))
    frame_roll = np.zeros((n_pitches, n_frames), dtype=np.float32)
    onset_roll = np.zeros((n_pitches, n_frames), dtype=np.float32)

    for note in notes:
        if not (pitch_low <= note.pitch <= pitch_high):
            continue  # outside the modeled range (rare at the extremes)
        row = note.pitch - pitch_low
        start_frame = max(0, int(np.floor(note.start * fs)))
        end_frame = min(n_frames, int(np.ceil(note.end * fs)))
        if end_frame <= start_frame:
            end_frame = min(n_frames, start_frame + 1)  # keep very short notes visible
        frame_roll[row, start_frame:end_frame] = 1.0
        onset_roll[row, start_frame] = 1.0

    return frame_roll, onset_roll
