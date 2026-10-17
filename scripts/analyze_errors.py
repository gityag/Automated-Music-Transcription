"""
Where do the models fail? Recall of reference notes broken down by note
duration, register, polyphony and (when MIDI is available) whether the
sustain pedal extended the note, for one or more checkpoints scored on the
same pieces. Also: how often onset-matched notes end too early or too late.

The question it answers for the paper: the CNN alone sees +-15 frames
(+-240 ms); if the per-pitch GRU's gain comes from memory over time, its
offset gain should grow with note duration and with pedal extension.

Uncertainty: every recall comes with a 95% interval from a bootstrap over
PIECES (notes within a piece are correlated), and each later --ckpt is
compared bin by bin with the FIRST one (paired, same resampled pieces). With
--midi-dir, the pedal penalty (recall of non-extended minus pedal-extended
notes) is also given per duration bin, with an interval.

Thresholds: each model's tuned defaults (amt.inference.default_thresholds),
i.e. the same decoding as the reported results.

Usage (SMD, on a laptop; --midi-dir enables the pedal breakdown):
  python scripts/analyze_errors.py --dir data/processed/smd --midi-dir data/smd/midi \
      --ckpt run2=models/run2_best.pt --ckpt run3=models/run3_best.pt \
      --ckpt run4=models/run4_best.pt --out results/analysis_smd.json
  (MAESTRO: --dir $CACHE/test --midi-dir $MAESTRO_ROOT, MIDI found recursively)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from amt.data.cache import load_piece
from amt.data.midi import Note, load_notes
from amt.evaluation.analysis import (DURATION_BINS, POLYPHONY_BINS, REGISTER_BINS,
                                     bootstrap_breakdown, bootstrap_group_gap,
                                     bootstrap_paired_diff, match_details, note_features,
                                     offset_error_summary, pedal_extended_mask)
from amt.evaluation.decode import decode_notes
from amt.inference import default_thresholds, load_checkpoint
from amt.models.cnn import predict_piece

DUR_LABELS = ["<0.25 s", "0.25-0.5 s", "0.5-1 s", "1-2 s", ">=2 s"]
REG_LABELS = ["low (A0-B2)", "mid (C3-B4)", "high (C5-C8)"]
POLY_LABELS = ["1-2", "3-4", "5-6", "7+"]


def fmt_ci(r: dict, key: str) -> str:
    if r.get(key) is None:
        return "  -              "
    return f"{r[key]:.3f} [{r[key + '_lo']:.3f}, {r[key + '_hi']:.3f}]"


def fmt_diff(r: dict) -> str:
    if r["diff"] is None:
        return "  -"
    return f"{r['diff'] * 100:+6.1f} pts [{r['lo'] * 100:+.1f}, {r['hi'] * 100:+.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder of cached .npz files (with reference notes)")
    ap.add_argument("--ckpt", action="append", required=True, help="NAME=PATH, repeatable")
    ap.add_argument("--midi-dir", default=None, help="MIDI files with the same stems, for the pedal breakdown")
    ap.add_argument("--max-pieces", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples over pieces")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(Path(args.dir).rglob("*.npz"))[: args.max_pieces]
    pieces = [load_piece(p, with_notes=True) for p in paths]
    refs = [[Note(int(p_), float(s), float(e), 100)
             for p_, s, e in zip(pc.ref_pitches, pc.ref_onsets, pc.ref_offsets)] for pc in pieces]

    pedal = None
    if args.midi_dir:
        # searched recursively, .mid or .midi (MAESTRO keeps .midi files in year folders)
        midis = {m.stem: m for m in Path(args.midi_dir).rglob("*") if m.suffix.lower() in (".mid", ".midi")}
        missing = [pc.stem for pc in pieces if pc.stem not in midis]
        if missing:
            print(f"warning: no MIDI for {len(missing)} pieces (counted as not pedal-extended), "
                  f"e.g. {missing[:2]}")
        pedal = []
        for pc, ref in zip(pieces, refs):
            midi = midis.get(pc.stem)
            pedal.append(pedal_extended_mask(ref, load_notes(str(midi))) if midi is not None
                         else np.zeros(len(ref), dtype=bool))
    feats = [note_features(ref, None if pedal is None else pedal[i]) for i, ref in enumerate(refs)]
    pooled = {k: np.concatenate([f[k] for f in feats]) for k in feats[0]}
    piece_ids = np.concatenate([np.full(len(ref), i) for i, ref in enumerate(refs)]).astype(int)
    boot = {"n_resamples": args.n_boot, "seed": 0}

    def bd(values, hits, bins, labels, mask=None):
        return bootstrap_breakdown(values, piece_ids, hits, bins, labels, mask=mask, **boot)
    print(f"{len(pieces)} pieces, {len(pooled['duration'])} reference notes"
          + (f", {pooled['pedal_extended'].mean():.0%} pedal-extended" if pedal is not None else ""))

    result = {"dir": args.dir, "n_pieces": len(pieces), "n_notes": int(len(pooled["duration"])),
              "bootstrap": {"unit": "piece", **boot}, "models": {}, "paired_vs_first": {}}
    all_hits = {}
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
        all_hits[name] = hits
        overall = bd(np.zeros(len(piece_ids)), hits, [0.0, 1.0], ["all"])[0]
        m = {"thresholds": thr,
             "overall": {k: overall[k] for k in overall if k.startswith("onset")},
             "by_duration": bd(pooled["duration"], hits, DURATION_BINS, DUR_LABELS),
             "by_register": bd(pooled["pitch"], hits, REGISTER_BINS, REG_LABELS),
             "by_polyphony": bd(pooled["polyphony"], hits, POLYPHONY_BINS, POLY_LABELS),
             "offset_errors": offset_error_summary(d["offset_err"], pooled["duration"])}
        if pedal is not None:
            ext = pooled["pedal_extended"]
            m["by_pedal"] = bd(ext.astype(float), hits, [0, 0.5, 1.5], ["not extended", "pedal-extended"])
            # pedal-extended notes are also longer: split each duration bin by pedal
            # status so the two effects can be told apart
            for flag, tag in ((False, "not_extended"), (True, "pedal_extended")):
                m[f"by_duration_{tag}"] = bd(pooled["duration"], hits, DURATION_BINS, DUR_LABELS,
                                             mask=ext == flag)
            # pedal penalty per duration bin: recall(not extended) - recall(extended)
            m["pedal_gap_by_duration"] = bootstrap_group_gap(
                pooled["duration"], piece_ids, d["oo_hit"], ext, DURATION_BINS, DUR_LABELS, **boot)
        result["models"][name] = m
        print(f"\n== {name} (frame_thr {thr['frame_thr']}, onset_thr {thr['onset_thr']}) ==")
        for title in ("by_duration", "by_register", "by_polyphony", "by_pedal",
                      "by_duration_not_extended", "by_duration_pedal_extended"):
            if title not in m:
                continue
            print(f"  {title}:")
            for r in m[title]:
                print(f"    {r['bin']:16s} n={r['n']:6d}  onset R {fmt_ci(r, 'onset_recall')}"
                      f"  on+off R {fmt_ci(r, 'onset_offset_recall')}")
        if "pedal_gap_by_duration" in m:
            print("  pedal penalty on on+off recall (not extended - extended):")
            for r in m["pedal_gap_by_duration"]:
                print(f"    {r['bin']:16s} n={r['n_rest']:6d}/{r['n_group']:<6d} {fmt_diff(r)}")
        oe = m["offset_errors"]
        if oe["n_onset_matched"]:
            print(f"  offsets of {oe['n_onset_matched']} onset-matched notes: too early {oe['too_early']:.1%}, "
                  f"too late {oe['too_late']:.1%}, median error {oe['median_error_s']*1000:+.0f} ms")
        else:
            print("  no onset-matched notes")

    # every later model vs the first one, same reference notes, paired resampling
    names = list(all_hits)
    for name in names[1:]:
        base = names[0]
        pair = {}
        for key, (vals, bins, labels) in {"by_duration": (pooled["duration"], DURATION_BINS, DUR_LABELS),
                                          "by_register": (pooled["pitch"], REGISTER_BINS, REG_LABELS),
                                          "by_polyphony": (pooled["polyphony"], POLYPHONY_BINS, POLY_LABELS)}.items():
            pair[key] = {h: bootstrap_paired_diff(vals, piece_ids, all_hits[base][h], all_hits[name][h],
                                                  bins, labels, **boot)
                         for h in ("onset_recall", "onset_offset_recall")}
        result["paired_vs_first"][f"{name}-{base}"] = pair
        print(f"\n== {name} - {base}: on+off recall by duration (paired, 95% CI) ==")
        for r in pair["by_duration"]["onset_offset_recall"]:
            print(f"    {r['bin']:16s} n={r['n']:6d}  {fmt_diff(r)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
