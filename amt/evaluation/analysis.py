"""
Note-level error analysis: which reference notes a system gets right,
broken down by note properties (duration, register, polyphony, sustain
pedal). Pure NumPy + mir_eval, same matching rules as note_metrics.

Everything here is recall-side: each REFERENCE note is labelled hit or
miss, so a breakdown by a property of the reference note is well defined
(precision-side breakdowns would need a property of the estimated note).
"""
from __future__ import annotations

import mir_eval
import numpy as np

from amt.data.midi import Note

# Bins used in the paper's tables (left-closed, right-open).
DURATION_BINS = [0.0, 0.25, 0.5, 1.0, 2.0, np.inf]           # seconds
REGISTER_BINS = [21, 48, 72, 109]                              # A0-B2 | C3-B4 | C5-C8
POLYPHONY_BINS = [1, 3, 5, 7, np.inf]                          # notes sounding at the onset


def match_details(ref: list[Note], est: list[Note], onset_tolerance: float = 0.05,
                  offset_ratio: float = 0.2) -> dict:
    """Per reference note: onset hit, onset+offset hit, and (for onset hits)
    the signed offset error est_end - ref_end in seconds (NaN otherwise)."""
    n = len(ref)
    onset_hit = np.zeros(n, dtype=bool)
    oo_hit = np.zeros(n, dtype=bool)
    offset_err = np.full(n, np.nan)
    by_pitch_ref: dict[int, list[int]] = {}
    by_pitch_est: dict[int, list[int]] = {}
    for i, r in enumerate(ref):
        by_pitch_ref.setdefault(r.pitch, []).append(i)
    for j, e in enumerate(est):
        by_pitch_est.setdefault(e.pitch, []).append(j)

    def arrays(notes, idx):
        iv = np.array([[notes[k].start, max(notes[k].end, notes[k].start + 1e-3)] for k in idx])
        hz = mir_eval.util.midi_to_hz(np.array([notes[k].pitch for k in idx], dtype=float))
        return iv, hz

    for pitch, ri in by_pitch_ref.items():
        ei = by_pitch_est.get(pitch)
        if not ei:
            continue
        r_iv, r_hz = arrays(ref, ri)
        e_iv, e_hz = arrays(est, ei)
        for a, b in mir_eval.transcription.match_notes(
                r_iv, r_hz, e_iv, e_hz, onset_tolerance=onset_tolerance,
                pitch_tolerance=50.0, offset_ratio=None):
            onset_hit[ri[a]] = True
            offset_err[ri[a]] = est[ei[b]].end - ref[ri[a]].end
        for a, _ in mir_eval.transcription.match_notes(
                r_iv, r_hz, e_iv, e_hz, onset_tolerance=onset_tolerance,
                pitch_tolerance=50.0, offset_ratio=offset_ratio):
            oo_hit[ri[a]] = True
    return {"onset_hit": onset_hit, "oo_hit": oo_hit, "offset_err": offset_err}


def note_features(ref: list[Note], pedal_extended: np.ndarray | None = None) -> dict:
    """Properties of each reference note used for the breakdowns."""
    starts = np.array([r.start for r in ref])
    ends = np.array([r.end for r in ref])
    # polyphony at onset: reference notes sounding at that instant (incl. itself)
    poly = np.empty(len(ref), dtype=int)
    for i in range(len(ref)):
        poly[i] = int(np.sum((starts <= starts[i]) & (ends > starts[i])))
    feats = {"duration": ends - starts,
             "pitch": np.array([r.pitch for r in ref]),
             "polyphony": poly}
    if pedal_extended is not None:
        feats["pedal_extended"] = np.asarray(pedal_extended, dtype=bool)
    return feats


def pedal_extended_mask(ref: list[Note], raw: list[Note], tol: float = 1e-3) -> np.ndarray:
    """True for reference notes whose end was pushed out by the sustain pedal:
    the raw (un-extended) note with the same pitch and onset ends earlier."""
    raw_end: dict[tuple[int, int], float] = {}
    for r in raw:
        key = (r.pitch, int(round(r.start / tol)))
        raw_end[key] = max(raw_end.get(key, -np.inf), r.end)
    out = np.zeros(len(ref), dtype=bool)
    for i, r in enumerate(ref):
        e = raw_end.get((r.pitch, int(round(r.start / tol))))
        out[i] = e is not None and r.end > e + tol
    return out


def breakdown(values: np.ndarray, hits: dict[str, np.ndarray], bins, labels=None) -> list[dict]:
    """Recall of each hit type within each bin of `values` (pooled notes)."""
    rows = []
    for k in range(len(bins) - 1):
        m = (values >= bins[k]) & (values < bins[k + 1])
        row = {"bin": labels[k] if labels else f"[{bins[k]}, {bins[k + 1]})", "n": int(m.sum())}
        for name, h in hits.items():
            row[name] = float(h[m].mean()) if m.any() else None
        rows.append(row)
    return rows


def offset_error_summary(offset_err: np.ndarray, durations: np.ndarray, offset_ratio: float = 0.2) -> dict:
    """For onset-matched notes: how often the predicted end is too early /
    too late beyond the onset+offset tolerance, and the median signed error."""
    m = ~np.isnan(offset_err)
    err, dur = offset_err[m], durations[m]
    tol = np.maximum(offset_ratio * dur, 0.05)
    return {"n_onset_matched": int(m.sum()),
            "too_early": float(np.mean(err < -tol)) if m.any() else None,
            "too_late": float(np.mean(err > tol)) if m.any() else None,
            "median_error_s": float(np.median(err)) if m.any() else None}
