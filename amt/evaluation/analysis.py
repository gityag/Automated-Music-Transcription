"""
Note-level error analysis: which reference notes a system gets right,
broken down by note properties (duration, register, polyphony, sustain
pedal). Pure NumPy + mir_eval, same matching rules as note_metrics.

Uncertainty: notes within a piece are not independent, so the intervals
here resample PIECES (cluster bootstrap) and recompute the pooled recall of
each bin from the resampled pieces' note counts. All bootstrap helpers use
the same resampling for a given (n_pieces, n_resamples, seed), so intervals
for different models and differences between them are paired.

Everything here is recall-side: each REFERENCE note is labelled hit or
miss, so a breakdown by a property of the reference note is well defined
(precision-side breakdowns would need a property of the estimated note).
"""
from __future__ import annotations

import warnings

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


# ---------------------------------------------------------------- uncertainty

def _bin_index(values: np.ndarray, bins) -> np.ndarray:
    """Bin of each value (left-closed, right-open); -1 outside all bins."""
    b = np.digitize(np.asarray(values, dtype=float), bins) - 1
    b[(b < 0) | (b >= len(bins) - 1)] = -1
    return b


def _piece_counts(values, piece_ids, n_pieces: int, bins, hit, mask=None):
    """(hits, n): per-piece, per-bin counts, each of shape (n_pieces, n_bins)."""
    b = _bin_index(values, bins)
    keep = b >= 0
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    hits = np.zeros((n_pieces, len(bins) - 1))
    n = np.zeros_like(hits)
    np.add.at(n, (piece_ids[keep], b[keep]), 1.0)
    np.add.at(hits, (piece_ids[keep], b[keep]), np.asarray(hit, dtype=float)[keep])
    return hits, n


def resample_weights(n_pieces: int, n_resamples: int = 2000, seed: int = 0) -> np.ndarray:
    """(n_resamples, n_pieces): how often each piece is drawn in each resample."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_pieces, size=(n_resamples, n_pieces))
    w = np.zeros((n_resamples, n_pieces))
    np.add.at(w, (np.repeat(np.arange(n_resamples), n_pieces), idx.ravel()), 1.0)
    return w


def _ratio(w, hits, n):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w @ hits) / (w @ n)            # (n_resamples, n_bins), NaN if a bin is empty


def _interval(samples: np.ndarray, alpha: float):
    with warnings.catch_warnings():          # an empty bin is all-NaN; it is reported as None
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanquantile(samples, [alpha / 2, 1 - alpha / 2], axis=0)
    return lo, hi


def bootstrap_breakdown(values, piece_ids, hits: dict[str, np.ndarray], bins, labels=None,
                        mask=None, n_resamples: int = 2000, alpha: float = 0.05,
                        seed: int = 0) -> list[dict]:
    """Like breakdown(), plus a 95% piece-level interval for each recall:
    keys <name>, <name>_lo, <name>_hi. Notes outside `mask` are ignored."""
    piece_ids = np.asarray(piece_ids)
    n_pieces = int(piece_ids.max()) + 1
    w = resample_weights(n_pieces, n_resamples, seed)
    rows = None
    for name, hit in hits.items():
        h, n = _piece_counts(values, piece_ids, n_pieces, bins, hit, mask)
        point = h.sum(0) / np.maximum(n.sum(0), 1)
        lo, hi = _interval(_ratio(w, h, n), alpha)
        if rows is None:
            rows = [{"bin": labels[k] if labels else f"[{bins[k]}, {bins[k + 1]})",
                     "n": int(n[:, k].sum())} for k in range(len(bins) - 1)]
        for k, r in enumerate(rows):
            empty = r["n"] == 0
            r[name] = None if empty else float(point[k])
            r[f"{name}_lo"] = None if empty else float(lo[k])
            r[f"{name}_hi"] = None if empty else float(hi[k])
    return rows


def _diff_rows(est_a, est_b, w, bins, labels, alpha, n_counts):
    (ha, na), (hb, nb) = est_a, est_b
    point = hb.sum(0) / np.maximum(nb.sum(0), 1) - ha.sum(0) / np.maximum(na.sum(0), 1)
    samples = _ratio(w, hb, nb) - _ratio(w, ha, na)
    lo, hi = _interval(samples, alpha)
    rows = []
    for k in range(len(bins) - 1):
        empty = n_counts[k] == 0
        col = samples[:, k]
        col = col[~np.isnan(col)]
        rows.append({"bin": labels[k] if labels else f"[{bins[k]}, {bins[k + 1]})",
                     "n": int(n_counts[k]),
                     "diff": None if empty else float(point[k]),
                     "lo": None if empty else float(lo[k]),
                     "hi": None if empty else float(hi[k]),
                     "p_le_zero": None if empty or not len(col) else float(np.mean(col <= 0))})
    return rows


def bootstrap_paired_diff(values, piece_ids, hit_a, hit_b, bins, labels=None,
                          n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0) -> list[dict]:
    """Per bin: recall(b) - recall(a) for two systems on the SAME reference
    notes, with a paired piece-level interval and p_le_zero (the share of
    resamples where b is not better)."""
    piece_ids = np.asarray(piece_ids)
    n_pieces = int(piece_ids.max()) + 1
    w = resample_weights(n_pieces, n_resamples, seed)
    a = _piece_counts(values, piece_ids, n_pieces, bins, hit_a)
    b = _piece_counts(values, piece_ids, n_pieces, bins, hit_b)
    return _diff_rows(a, b, w, bins, labels, alpha, a[1].sum(0))


def bootstrap_group_gap(values, piece_ids, hit, group, bins, labels=None,
                        n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0) -> list[dict]:
    """Per bin: recall(notes NOT in group) - recall(notes in group) for one
    system, e.g. the pedal penalty within each duration bin."""
    piece_ids = np.asarray(piece_ids)
    group = np.asarray(group, dtype=bool)
    n_pieces = int(piece_ids.max()) + 1
    w = resample_weights(n_pieces, n_resamples, seed)
    inside = _piece_counts(values, piece_ids, n_pieces, bins, hit, group)
    outside = _piece_counts(values, piece_ids, n_pieces, bins, hit, ~group)
    n_in, n_out = inside[1].sum(0), outside[1].sum(0)
    rows = _diff_rows(inside, outside, w, bins, labels, alpha, np.minimum(n_in, n_out))
    for k, r in enumerate(rows):
        r["n"], r["n_group"], r["n_rest"] = int(n_in[k] + n_out[k]), int(n_in[k]), int(n_out[k])
    return rows
