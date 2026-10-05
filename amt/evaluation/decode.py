"""
Turning model probabilities into notes, and scoring them.

Pure NumPy (no torch), so it is testable anywhere.

Two decoding modes, chosen by whether the model has an onset head:
- frame-only: a note starts where the thresholded frame probability
  rises from 0 to 1 and ends where it falls back.
- with onset head: a note starts at a local maximum of the onset
  probability (above onset_thr) and ends where the frame probability
  drops. Frame activity without an onset peak is ignored, which is what
  lets two repeated notes on one pitch be told apart while the frame
  head stays "on".
Both drop notes shorter than min_frames (about 50 ms at 3 frames), which
are overwhelmingly spurious flicker.
"""
from __future__ import annotations

import numpy as np

from amt.data.midi import Note, piano_roll_to_notes
from amt.evaluation.metrics import frame_metrics, note_metrics


def decode_notes(
    frame_probs: np.ndarray,
    onset_probs: np.ndarray | None = None,
    fs: float = 62.5,
    pitch_low: int = 21,
    frame_thr: float = 0.5,
    onset_thr: float = 0.5,
    min_frames: int = 3,
) -> list[Note]:
    frame_on = frame_probs >= frame_thr
    if onset_probs is None:
        onset = np.zeros_like(frame_on)
        onset[:, 0] = frame_on[:, 0]
        onset[:, 1:] = frame_on[:, 1:] & ~frame_on[:, :-1]
    else:
        p = onset_probs
        left = np.pad(p, ((0, 0), (1, 0)), constant_values=-1.0)[:, :-1]
        right = np.pad(p, ((0, 0), (0, 1)), constant_values=-1.0)[:, 1:]
        onset = (p >= onset_thr) & (p >= left) & (p > right)
        frame_on = frame_on | onset  # a note must be "on" at its own onset frame

    notes = piano_roll_to_notes(
        frame_on.astype(np.float32), onset.astype(np.float32), fs=fs, pitch_low=pitch_low
    )
    min_len = min_frames / fs
    return [n for n in notes if (n.end - n.start) >= min_len - 1e-9]


def _macro(dicts: list[dict]) -> dict:
    return {k: float(np.mean([d[k] for d in dicts])) for k in dicts[0]}


def score_predictions(
    pieces,
    preds: list[tuple[np.ndarray, np.ndarray | None]],
    fs: float = 62.5,
    pitch_low: int = 21,
    frame_thr: float = 0.5,
    onset_thr: float = 0.5,
    min_frames: int = 3,
) -> dict:
    """Macro-average (over pieces) frame and note metrics, the same
    convention scripts/run_baselines.py uses, so the numbers are
    directly comparable. Pieces must have been loaded with_notes=True."""
    frame_scores, note_scores = [], []
    for piece, (fprob, oprob) in zip(pieces, preds):
        ref_notes = [
            Note(pitch=int(p), start=float(s), end=float(e), velocity=100)
            for p, s, e in zip(piece.ref_pitches, piece.ref_onsets, piece.ref_offsets)
        ]
        est_notes = decode_notes(fprob, oprob, fs=fs, pitch_low=pitch_low, frame_thr=frame_thr,
                                 onset_thr=onset_thr, min_frames=min_frames)
        frame_scores.append(vars(frame_metrics(piece.frame, (fprob >= frame_thr).astype(np.float32))))
        note_scores.append(vars(note_metrics(ref_notes, est_notes)))
    return {"n_pieces": len(pieces), "frame": _macro(frame_scores), "note": _macro(note_scores)}