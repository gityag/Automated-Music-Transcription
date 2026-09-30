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


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def note_metrics(
    ref_notes: list[Note],
    est_notes: list[Note],
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
    pitch_tolerance_cents: float = 50.0,
) -> NoteMetrics:
    """
    Onset-only and onset+offset precision/recall/F1, equivalent to
    mir_eval.transcription.precision_recall_f1_overlap but computed
    per pitch.

    Why per pitch: a note can only match a note of (nearly) the same
    pitch, and MIDI pitches are 100 cents apart, so with
    pitch_tolerance_cents < 100 the bipartite matching decomposes
    exactly into independent per-pitch problems. Summing matched
    counts over pitches and computing P/R/F1 from the totals gives the
    same numbers as one global matching, at a fraction of the cost --
    which removes the need for any note-count cap (mir_eval's global
    matching is what used to blow up on dense or over-triggering
    predictions).

    onset_tolerance: seconds a predicted onset may differ from the
    reference (AMT-literature default: 50 ms).
    offset_ratio: for the onset+offset metric, the predicted offset
    must fall within max(offset_ratio * ref_duration, 0.05 s) of the
    reference offset. Onset-only is computed with offset_ratio=None.
    """
    if pitch_tolerance_cents >= 100.0:
        raise ValueError(
            "per-pitch decomposition is only exact for pitch_tolerance_cents < 100"
        )

    n_ref, n_est = len(ref_notes), len(est_notes)
    if n_ref == 0 or n_est == 0:
        return NoteMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    ref_by_pitch: dict[int, list[Note]] = {}
    est_by_pitch: dict[int, list[Note]] = {}
    for n in ref_notes:
        ref_by_pitch.setdefault(n.pitch, []).append(n)
    for n in est_notes:
        est_by_pitch.setdefault(n.pitch, []).append(n)

    onset_matches = 0
    oo_matches = 0
    for pitch, refs in ref_by_pitch.items():
        ests = est_by_pitch.get(pitch)
        if not ests:
            continue
        ref_int, ref_p = _notes_to_mir_eval_format(refs)
        est_int, est_p = _notes_to_mir_eval_format(ests)
        onset_matches += len(mir_eval.transcription.match_notes(
            ref_int, ref_p, est_int, est_p,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=pitch_tolerance_cents,
            offset_ratio=None,
        ))
        oo_matches += len(mir_eval.transcription.match_notes(
            ref_int, ref_p, est_int, est_p,
            onset_tolerance=onset_tolerance,
            pitch_tolerance=pitch_tolerance_cents,
            offset_ratio=offset_ratio,
        ))

    onset_p, onset_r = onset_matches / n_est, onset_matches / n_ref
    oo_p, oo_r = oo_matches / n_est, oo_matches / n_ref
    return NoteMetrics(
        onset_precision=onset_p,
        onset_recall=onset_r,
        onset_f1=_f1(onset_p, onset_r),
        onset_offset_precision=oo_p,
        onset_offset_recall=oo_r,
        onset_offset_f1=_f1(oo_p, oo_r),
    )