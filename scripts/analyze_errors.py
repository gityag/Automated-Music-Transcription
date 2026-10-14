"""
Where do the models fail? Recall of reference notes broken down by note
duration, register, polyphony and (when MIDI is available) whether the
sustain pedal extended the note, for one or more checkpoints scored on the
same pieces. Also: how often onset-matched notes end too early or too late.

The question it answers for the paper: the CNN alone sees +-15 frames
(+-240 ms); if the per-pitch GRU's gain comes from memory over time, its
offset gain should grow with note duration and with pedal extension.

Thresholds: each model's tuned defaults (amt.inference.default_thresholds),
i.e. the same decoding as the reported results.

Usage (SMD, on a laptop; --midi-dir enables the pedal breakdown):
  python scripts/analyze_errors.py --dir data/processed/smd --midi-dir data/smd/midi \
      --ckpt run2=models/run2_best.pt --ckpt run3=models/run3_best.pt \
      --ckpt run4=models/run4_best.pt --out results/analysis_smd.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from amt.data.cache import load_piece
from amt.data.midi import Note, load_notes
from amt.evaluation.analysis import (DURATION_BINS, POLYPHONY_BINS, REGISTER_BINS, breakdown,
                                     match_details, note_features, offset_error_summary,
                                     pedal_extended_mask)
from amt.evaluation.decode import decode_notes
from amt.inference import default_thresholds, load_checkpoint
from amt.models.cnn import predict_piece

DUR_LABELS = ["<0.25 s", "0.25-0.5 s", "0.5-1 s", "1-2 s", ">=2 s"]
REG_LABELS = ["low (A0-B2)", "mid (C3-B4)", "high (C5-C8)"]
POLY_LABELS = ["1-2", "3-4", "5-6", "7+"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder of cached .npz files (with reference notes)")
    ap.add_argument("--ckpt", action="append", required=True, help="NAME=PATH, repeatable")
    ap.add_argument("--midi-dir", default=None, help="MIDI files with the same stems, for the pedal breakdown")
    ap.add_argument("--max-pieces", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(Path(args.dir).rglob("*.npz"))[: args.max_pieces]
    pieces = [load_piece(p, with_notes=True) for p in paths]
    refs = [[Note(int(p_), float(s), float(e), 100)
             for p_, s, e in zip(pc.ref_pitches, pc.ref_onsets, pc.ref_offsets)] for pc in pieces]

    pedal = None
    if args.midi_dir:
        pedal = []
        for pc, ref in zip(pieces, refs):
            midi = Path(args.midi_dir) / f"{pc.stem}.mid"
            pedal.append(pedal_extended_mask(ref, load_notes(str(midi))) if midi.exists()
                         else np.zeros(len(ref), dtype=bool))
    feats = [note_features(ref, None if pedal is None else pedal[i]) for i, ref in enumerate(refs)]
    pooled = {k: np.concatenate([f[k] for f in feats]) for k in feats[0]}
    print(f"{len(pieces)} pieces, {len(pooled['duration'])} reference notes"
          + (f", {pooled['pedal_extended'].mean():.0%} pedal-extended" if pedal is not None else ""))

    result = {"dir": args.dir, "n_pieces": len(pieces), "n_notes": int(len(pooled["duration"])),
              "models": {}}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        model, cfg, device = load_checkpoint(path)
        thr = default_thresholds(model)
        fs = cfg.audio.sample_rate / cfg.cqt.hop_length
        details = []
        for pc, ref in zip(pieces, refs):
            fprob, oprob = predict_piece(model, pc.cqt, device=device)
            est = decode_notes(fprob, oprob, fs=fs, pitch_low=cfg.labels.pitch_low, **thr)
            details.append(match_details(ref, est))
        d = {k: np.concatenate([x[k] for x in details]) for k in details[0]}
        hits = {"onset_recall": d["onset_hit"], "onset_offset_recall": d["oo_hit"]}
        m = {"thresholds": thr,
             "overall": {"onset_recall": float(d["onset_hit"].mean()),
                         "onset_offset_recall": float(d["oo_hit"].mean())},
             "by_duration": breakdown(pooled["duration"], hits, DURATION_BINS, DUR_LABELS),
             "by_register": breakdown(pooled["pitch"], hits, REGISTER_BINS, REG_LABELS),
             "by_polyphony": breakdown(pooled["polyphony"], hits, POLYPHONY_BINS, POLY_LABELS),
             "offset_errors": offset_error_summary(d["offset_err"], pooled["duration"])}
        if pedal is not None:
            m["by_pedal"] = breakdown(pooled["pedal_extended"].astype(float), hits, [0, 0.5, 1.5],
                                      ["not extended", "pedal-extended"])
            # pedal-extended notes are also longer: split each duration bin by pedal
            # status so the two effects can be told apart
            for flag, tag in ((False, "not_extended"), (True, "pedal_extended")):
                sel = pooled["pedal_extended"] == flag
                m[f"by_duration_{tag}"] = breakdown(pooled["duration"][sel],
                                                    {k: v[sel] for k, v in hits.items()},
                                                    DURATION_BINS, DUR_LABELS)
        result["models"][name] = m
        print(f"\n== {name} (frame_thr {thr['frame_thr']}, onset_thr {thr['onset_thr']}) ==")
        for title in ("by_duration", "by_register", "by_polyphony", "by_pedal",
                      "by_duration_not_extended", "by_duration_pedal_extended"):
            if title not in m:
                continue
            print(f"  {title}:")
            for r in m[title]:
                on = "  -  " if r["onset_recall"] is None else f"{r['onset_recall']:.3f}"
                oo = "  -  " if r["onset_offset_recall"] is None else f"{r['onset_offset_recall']:.3f}"
                print(f"    {r['bin']:16s} n={r['n']:6d}  onset R {on}  on+off R {oo}")
        oe = m["offset_errors"]
        if oe["n_onset_matched"]:
            print(f"  offsets of {oe['n_onset_matched']} onset-matched notes: too early {oe['too_early']:.1%}, "
                  f"too late {oe['too_late']:.1%}, median error {oe['median_error_s']*1000:+.0f} ms")
        else:
            print("  no onset-matched notes")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
