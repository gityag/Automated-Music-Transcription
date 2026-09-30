"""
Score the silence and spectral-peak baselines against real SMD data.

Requires scripts/cache_smd_features.py to have already been run (this
reads data/processed/smd/*.npz, not the raw audio/MIDI). Ground-truth
notes are the continuous MIDI notes cached alongside the features
(ref_onsets/ref_offsets/ref_pitches); predicted notes are decoded from
the predicted rolls via piano_roll_to_notes.

Note-level scoring uses amt.evaluation.metrics.note_metrics, which matches
notes per pitch (exactly equivalent to global mir_eval matching, but fast
enough that no note-count cap is needed -- every song is scored).

Usage:
    python3 scripts/run_baselines.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from amt.baselines import silence_prediction, spectral_peak_prediction
from amt.config import Config
from amt.data.midi import Note, piano_roll_to_notes
from amt.evaluation.metrics import frame_metrics, note_metrics
from amt.features.cqt import frame_rate

CACHE_DIR = Path("data/processed/smd")
RESULTS_DIR = Path("results")
SPECTRAL_THRESHOLD_DB = -40.0


def macro_average(dicts: list[dict]) -> dict:
    """Average each numeric field across a list of same-shaped dicts."""
    keys = dicts[0].keys()
    return {k: float(np.mean([d[k] for d in dicts])) for k in keys}


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

        # Reference notes: continuous, pedal-extended times straight from the
        # MIDI (cached by cache_smd_features.py), not decoded from the
        # frame-quantized roll. Predictions are frame-quantized by nature;
        # the reference should not be.
        ref_notes = [
            Note(pitch=int(p), start=float(s), end=float(e), velocity=100)
            for p, s, e in zip(d["ref_pitches"], d["ref_onsets"], d["ref_offsets"])
        ]

        # --- silence baseline ---
        sil_frame_roll, sil_onset_roll = silence_prediction(n_pitches, n_frames)
        silence_frame_scores.append(vars(frame_metrics(ref_frame_roll, sil_frame_roll)))
        sil_notes = piano_roll_to_notes(sil_frame_roll, sil_onset_roll, fs=fs, pitch_low=cfg.labels.pitch_low)
        silence_note_scores.append(vars(note_metrics(ref_notes, sil_notes)))

        # --- spectral-peak baseline ---
        spec_frame_roll, spec_onset_roll = spectral_peak_prediction(cqt, threshold_db=SPECTRAL_THRESHOLD_DB)
        spectral_frame_scores.append(vars(frame_metrics(ref_frame_roll, spec_frame_roll)))
        spec_notes = piano_roll_to_notes(spec_frame_roll, spec_onset_roll, fs=fs, pitch_low=cfg.labels.pitch_low)
        spectral_note_scores.append(vars(note_metrics(ref_notes, spec_notes)))

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
            print("  note-level: no songs scored\n")

    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()