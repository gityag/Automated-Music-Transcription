"""
Evaluation metrics for piano transcription.

Two kinds of metric, matching the standard AMT evaluation split:

- Frame-level: precision/recall/F1 on the binary frame_roll/onset_roll
  arrays your pipeline already produces (amt.data.midi.notes_to_piano_roll).
  These arrays are already time/pitch aligned between reference and
  estimate, so this is plain NumPy -- no need for mir_eval's multipitch
  module, which is built for a different input shape (per-frame lists
  of active frequencies, not a fixed binary grid).

- Note-level: onset-only and onset+offset precision/recall/F1 via
  mir_eval.transcription, the standard evaluator used in the AMT
  literature (e.g. Onsets and Frames, Basic Pitch). Operates on
  amt.data.midi.Note lists, not on the piano-roll arrays -- note-level
  matching needs continuous onset/offset times, not frame-quantized ones.
"""
from __future__ import annotations

from dataclasses import dataclass

import mir_eval
import numpy as np

from amt.data.midi import Note


@dataclass(frozen=True)
class FrameMetrics:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class NoteMetrics:
    onset_precision: float
    onset_recall: float
    onset_f1: float
    onset_offset_precision: float
    onset_offset_recall: float
    onset_offset_f1: float


def frame_metrics(ref_roll: np.ndarray, est_roll: np.ndarray, threshold: float = 0.5) -> FrameMetrics:
    """
    Precision/recall/F1 over a binary (pitch, frame) grid.

    ref_roll and est_roll must be the same shape -- same pitch range,
    same frame count, same frame rate (this is a per-cell comparison,
    not a matching problem, so any misalignment silently produces
    meaningless numbers rather than an error; callers are responsible
    for using consistent cfg.cqt / cfg.labels settings on both sides).

    Convention for the empty-array edge case (no active cells in ref
    or est at all): if there is nothing to predict and nothing was
    predicted, precision/recall/F1 are all 1.0 (a perfect, if trivial,
    match). Otherwise a zero denominator yields 0.0, matching sklearn's
    default behavior.
    """
    if ref_roll.shape != est_roll.shape:
        raise ValueError(
            f"ref_roll shape {ref_roll.shape} != est_roll shape {est_roll.shape}"
        )

    ref = ref_roll >= threshold
    est = est_roll >= threshold

    tp = np.sum(ref & est)
    fp = np.sum(~ref & est)
    fn = np.sum(ref & ~est)

    if tp + fp == 0 and tp + fn == 0:
        return FrameMetrics(precision=1.0, recall=1.0, f1=1.0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return FrameMetrics(precision=float(precision), recall=float(recall), f1=float(f1))


def _notes_to_mir_eval_format(notes: list[Note]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of Note into the (intervals, pitches_hz) arrays
    mir_eval.transcription expects: intervals shape (N, 2) of
    [onset, offset] seconds, pitches shape (N,) in Hz.

    Returns properly-shaped empty arrays ((0, 2) and (0,)) for an empty
    note list, rather than a shape mir_eval would reject -- an empty
    prediction or an empty reference (e.g. a silent passage) is a real,
    valid case, not an error.
    """
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array([[n.start, n.end] for n in notes], dtype=float)
    pitches_hz = np.array([440.0 * 2 ** ((n.pitch - 69) / 12) for n in notes], dtype=float)
    return intervals, pitches_hz


def note_metrics(
    ref_notes: list[Note],
    est_notes: list[Note],
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
    pitch_tolerance_cents: float = 50.0,
) -> NoteMetrics:
    """
    Onset-only and onset+offset precision/recall/F1, via
    mir_eval.transcription.precision_recall_f1_overlap.

    onset_tolerance: seconds a predicted onset may differ from the
    reference and still count as a match (mir_eval/AMT-literature
    default: 50ms).
    offset_ratio: for the onset+offset metric, the predicted offset
    must fall within max(offset_ratio * ref_duration, 0.05s) of the
    reference offset (mir_eval's standard rule). Passing
    offset_ratio=None to mir_eval disables the offset requirement
    entirely, giving the onset-only metric.
    pitch_tolerance_cents: how close (in cents) an estimated pitch must
    be to the reference to count as the same note. 50 cents = a quarter
    semitone; exact MIDI-pitch-derived Hz values match at 0 cents, so
    this only matters once real (non-quantized) pitch estimates are
    involved.
    """
    ref_intervals, ref_pitches = _notes_to_mir_eval_format(ref_notes)
    est_intervals, est_pitches = _notes_to_mir_eval_format(est_notes)

    onset_p, onset_r, onset_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance_cents,
        offset_ratio=None,
    )
    oo_p, oo_r, oo_f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance_cents,
        offset_ratio=offset_ratio,
    )

    return NoteMetrics(
        onset_precision=float(onset_p),
        onset_recall=float(onset_r),
        onset_f1=float(onset_f1),
        onset_offset_precision=float(oo_p),
        onset_offset_recall=float(oo_r),
        onset_offset_f1=float(oo_f1),
    )
