"""
Score the silence and spectral-peak baselines against real SMD data.

Requires scripts/cache_smd_features.py to have already been run (this
reads data/processed/smd/*.npz, not the raw audio/MIDI). Ground-truth
Notes are decoded from the cached frame_roll/onset_roll via
piano_roll_to_notes, so both the "reference" and "prediction" sides of
note_metrics go through the exact same decoder -- if the decoder itself
had a systematic bias, this keeps it from silently favoring one side.

Safety cap: a poorly-tuned threshold (or, later, a poorly-trained
model) can produce a pathological number of tiny spurious "notes" on
real audio -- mir_eval's note-matching cost grows badly with note
count, and a single bad song can make the whole run hang rather than
fail loudly. MAX_NOTES_PER_SONG guards against that: if either side's
decoded note count exceeds it for a song, that song's note-level score
is skipped (with a printed warning) while frame-level scoring, which
is a fast vectorized array comparison independent of note count, still
runs normally.

Usage:
    python3 scripts/run_baselines.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from amt.baselines import silence_prediction, spectral_peak_prediction
from amt.config import Config
from amt.data.midi import piano_roll_to_notes
from amt.evaluation.metrics import frame_metrics, note_metrics
from amt.features.cqt import frame_rate

CACHE_DIR = Path("data/processed/smd")
RESULTS_DIR = Path("results")
SPECTRAL_THRESHOLD_DB = -40.0
MAX_NOTES_PER_SONG = 5000  # real piano pieces rarely exceed a few thousand notes


def macro_average(dicts: list[dict]) -> dict:
    """Average each numeric field across a list of same-shaped dicts."""
    keys = dicts[0].keys()
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


def safe_note_metrics(ref_notes, est_notes, stem: str, system: str) -> dict | None:
    """note_metrics, but returns None (and warns) instead of risking a
    very slow / hung mir_eval call on a pathological note count.

    An empty estimate (the silence baseline always has one) is always
    cheap to match against, however large the reference is -- matching
    against zero candidates is O(1), not a function of ref size. Real
    dense pieces (a 17-minute Liszt sonata can legitimately have
    ~12,000 notes) shouldn't lose their silence-baseline score just
    because they're long; the cap exists to catch a *predicted* side
    that has exploded into tens of thousands of spurious notes, not a
    long piece's genuinely large reference note count.
    """
    if len(est_notes) == 0:
        return vars(note_metrics(ref_notes, est_notes))
    if len(ref_notes) > MAX_NOTES_PER_SONG or len(est_notes) > MAX_NOTES_PER_SONG:
        print(
            f"  [skip note-level] {stem} / {system}: "
            f"{len(ref_notes)} ref notes, {len(est_notes)} est notes "
            f"(cap is {MAX_NOTES_PER_SONG})"
        )
        return None
    return vars(note_metrics(ref_notes, est_notes))


def main() -> None:
    cfg = Config()
    fs = frame_rate(cfg.audio.sample_rate, cfg.cqt.hop_length)

    cache_files = sorted(CACHE_DIR.glob("*.npz"))
    if not cache_files:
        raise SystemExit(
            f"No cached features found in {CACHE_DIR}. "
            "Run scripts/cache_smd_features.py first."
        )

    silence_frame_scores, silence_note_scores = [], []
    spectral_frame_scores, spectral_note_scores = [], []

    for path in cache_files:
        d = np.load(path)
        cqt = d["cqt"]
        ref_frame_roll = d["frame_roll"]
        ref_onset_roll = d["onset_roll"]
        n_pitches, n_frames = ref_frame_roll.shape
        stem = str(d["stem"])

        ref_notes = piano_roll_to_notes(
            ref_frame_roll, ref_onset_roll, fs=fs, pitch_low=cfg.labels.pitch_low
        )

        # --- silence baseline ---
        sil_frame_roll, sil_onset_roll = silence_prediction(n_pitches, n_frames)
        silence_frame_scores.append(vars(frame_metrics(ref_frame_roll, sil_frame_roll)))
        sil_notes = piano_roll_to_notes(sil_frame_roll, sil_onset_roll, fs=fs, pitch_low=cfg.labels.pitch_low)
        note_score = safe_note_metrics(ref_notes, sil_notes, stem, "silence")
        if note_score is not None:
            silence_note_scores.append(note_score)

        # --- spectral-peak baseline ---
        spec_frame_roll, spec_onset_roll = spectral_peak_prediction(cqt, threshold_db=SPECTRAL_THRESHOLD_DB)
        spectral_frame_scores.append(vars(frame_metrics(ref_frame_roll, spec_frame_roll)))
        spec_notes = piano_roll_to_notes(spec_frame_roll, spec_onset_roll, fs=fs, pitch_low=cfg.labels.pitch_low)
        note_score = safe_note_metrics(ref_notes, spec_notes, stem, "spectral_peak")
        if note_score is not None:
            spectral_note_scores.append(note_score)

    results = {
        "n_songs": len(cache_files),
        "spectral_threshold_db": SPECTRAL_THRESHOLD_DB,
        "silence": {
            "frame": macro_average(silence_frame_scores),
            "note": macro_average(silence_note_scores) if silence_note_scores else None,
            "n_songs_scored_for_notes": len(silence_note_scores),
        },
        "spectral_peak": {
            "frame": macro_average(spectral_frame_scores),
            "note": macro_average(spectral_note_scores) if spectral_note_scores else None,
            "n_songs_scored_for_notes": len(spectral_note_scores),
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "baselines_smd.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"\nScored {len(cache_files)} SMD songs\n")
    for system in ("silence", "spectral_peak"):
        r = results[system]
        print(f"{system}:")
        print(f"  frame F1: {r['frame']['f1']:.3f}  "
              f"(P={r['frame']['precision']:.3f}, R={r['frame']['recall']:.3f})")
        if r["note"] is not None:
            print(f"  note onset F1: {r['note']['onset_f1']:.3f}  "
                  f"(P={r['note']['onset_precision']:.3f}, R={r['note']['onset_recall']:.3f}), "
                  f"scored on {r['n_songs_scored_for_notes']}/{len(cache_files)} songs")
            print(f"  note onset+offset F1: {r['note']['onset_offset_f1']:.3f}\n")
        else:
            print(f"  note-level: all songs skipped (see warnings above)\n")

    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()