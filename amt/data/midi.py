"""
MIDI parsing and label construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pretty_midi


@dataclass(frozen=True)
class Note:
    pitch: int
    start: float
    end: float
    velocity: int


def load_notes(midi_path: str) -> list[Note]:
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
    n_frames: int,
    fs: int,
    pitch_low: int,
    pitch_high: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_pitches = pitch_high - pitch_low + 1
    frame_roll = np.zeros((n_pitches, n_frames), dtype=np.float32)
    onset_roll = np.zeros((n_pitches, n_frames), dtype=np.float32)

    for note in notes:
        if not (pitch_low <= note.pitch <= pitch_high):
            continue
        row = note.pitch - pitch_low
        start_frame = min(max(0, int(np.floor(note.start * fs))), n_frames - 1)
        end = min(note.end, n_frames / fs)
        end_frame = min(n_frames, int(np.ceil(end * fs)))
        if end_frame <= start_frame:
            end_frame = min(n_frames, start_frame + 1)
        frame_roll[row, start_frame:end_frame] = 1.0
        onset_roll[row, start_frame] = 1.0

    return frame_roll, onset_roll


def piano_roll_to_notes(
    frame_roll: np.ndarray,
    onset_roll: np.ndarray,
    fs: float,
    pitch_low: int,
    default_velocity: int = 100,
) -> list[Note]:
    """
    Inverse of notes_to_piano_roll: turn a (frame_roll, onset_roll) pair
    back into a list of Notes.

    Used two ways: (1) scoring a model's (or a baseline's) frame/onset
    predictions with amt.evaluation.metrics.note_metrics, which needs
    Note objects, not arrays; (2) decoding a cached ground-truth roll
    back into Notes for evaluation, so scoring code only has to handle
    one input shape.

    A new note starts at any frame where onset_roll is 1 -- this is
    what lets two re-struck notes on the same pitch with no gap between
    them decode as two separate notes rather than one long one, which a
    frame_roll alone (without onset_roll) couldn't distinguish. A note
    ends at the first frame after its onset where frame_roll drops to 0,
    or at the recording's end, or at the next onset for the same pitch
    (whichever comes first).

    Velocity isn't recoverable from a binary roll, so every decoded note
    gets `default_velocity`; note-level metrics never use velocity, so
    this is inert for scoring, only relevant if the notes are later
    exported to a MIDI file.
    """
    n_pitches, n_frames = frame_roll.shape
    notes: list[Note] = []

    for row in range(n_pitches):
        pitch = row + pitch_low
        onset_frames = np.where(onset_roll[row] >= 0.5)[0]
        for i, onset_frame in enumerate(onset_frames):
            next_onset = onset_frames[i + 1] if i + 1 < len(onset_frames) else n_frames
            end_frame = onset_frame + 1
            while end_frame < next_onset and frame_roll[row, end_frame] >= 0.5:
                end_frame += 1
            notes.append(
                Note(
                    pitch=pitch,
                    start=onset_frame / fs,
                    end=end_frame / fs,
                    velocity=default_velocity,
                )
            )

    return sorted(notes, key=lambda n: (n.start, n.pitch))
