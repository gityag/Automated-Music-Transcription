"""
Scoring notes produced by an EXTERNAL transcription system with exactly
our metric code, so its numbers sit in the same table as ours.

The external system brings its own decoding (thresholds, post-processing);
we only take its final notes. Note metrics are computed against the same
continuous, pedal-extended reference notes as our models. The frame metric
is computed by rasterising its notes onto our frame grid, which is the
fair way to compare against a system that outputs notes, not frames.
"""
from __future__ import annotations

import numpy as np

from amt.data.midi import Note, notes_to_piano_roll
from amt.evaluation.metrics import frame_metrics, note_metrics


def events_to_notes(events: list[dict]) -> list[Note]:
    """Kong et al.'s note events ({'onset_time', 'offset_time', 'midi_note',
    'velocity'}) -> our Note objects."""
    return [Note(pitch=int(e["midi_note"]), start=float(e["onset_time"]),
                 end=float(e["offset_time"]), velocity=int(e.get("velocity", 100)))
            for e in events]


def score_external_notes(est: list[Note], piece, fs: float = 62.5,
                         pitch_low: int = 21, pitch_high: int = 108) -> dict:
    """One piece -> {"stem", "frame", "note"}, the per-piece layout of
    score_predictions(..., per_piece=True)."""
    ref = [Note(pitch=int(p), start=float(s), end=float(e), velocity=100)
           for p, s, e in zip(piece.ref_pitches, piece.ref_onsets, piece.ref_offsets)]
    n_frames = piece.frame.shape[1]
    est_roll, _ = notes_to_piano_roll(est, n_frames=n_frames, fs=fs,
                                      pitch_low=pitch_low, pitch_high=pitch_high)
    return {"stem": piece.stem,
            "frame": vars(frame_metrics(piece.frame.astype(np.float32), est_roll)),
            "note": vars(note_metrics(ref, est))}
