"""
Score a trained checkpoint with frame- and note-level metrics.

Optionally tunes the decoding thresholds on a separate (validation)
folder first, then applies the chosen thresholds to the evaluation
folder -- never tune on the data you report.

The output JSON holds the macro averages, 95% bootstrap intervals over
pieces ("ci"), and every piece's scores ("per_piece"), so two models
scored on the same folder can later be compared piece by piece
(scripts/compare_runs.py).

Works on any folder of cached .npz files with the standard keys, e.g. a
MAESTRO split folder or data/processed/smd.

Examples:
  python scripts/evaluate_model.py --ckpt run1/best.pt \
      --dir $CACHE/test --tune-dir $CACHE/validation --out results/model_maestro_test.json
  python scripts/evaluate_model.py --ckpt run1/best.pt --dir data/processed/smd \
      --frame-thr 0.5 --out results/model_smd.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from amt.data.cache import load_piece
from amt.evaluation.decode import score_predictions
from amt.evaluation.stats import summarize
from amt.inference import load_checkpoint
from amt.models.cnn import predict_piece


def load_pieces(folder: str, max_pieces: int | None, seed: int = 21):
    paths = sorted(Path(folder).glob("*.npz"))
    random.Random(seed).shuffle(paths)
    if max_pieces:
        paths = paths[:max_pieces]
    return [load_piece(p, with_notes=True) for p in paths]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dir", required=True, help="folder of .npz files to report on")
    ap.add_argument("--tune-dir", default=None, help="folder to pick thresholds on (validation)")
    ap.add_argument("--tune-pieces", type=int, default=24)
    ap.add_argument("--max-pieces", type=int, default=None)
    ap.add_argument("--frame-thr", type=float, default=0.5)
    ap.add_argument("--onset-thr", type=float, default=0.5)
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--grid-frame", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7",
                    help="comma-separated frame thresholds to try when tuning")
    ap.add_argument("--grid-onset", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                    help="comma-separated onset thresholds to try (onset-head models only)")
    ap.add_argument("--tune-metric", default="onset_offset_f1",
                    choices=["onset_f1", "onset_offset_f1"],
                    help="note metric the threshold search maximizes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model, cfg, device = load_checkpoint(args.ckpt)
    ck_epoch = model.ckpt_epoch
    fs = cfg.audio.sample_rate / cfg.cqt.hop_length
    common = dict(fs=fs, pitch_low=cfg.labels.pitch_low, min_frames=args.min_frames)
    has_onset = model.onset_out is not None
    print(f"checkpoint epoch {ck_epoch}, onset head: {has_onset}, device: {device}")

    frame_thr, onset_thr = args.frame_thr, args.onset_thr
    tuning = []
    if args.tune_dir:
        tune = load_pieces(args.tune_dir, args.tune_pieces)
        tune_preds = [predict_piece(model, p.cqt, device=device) for p in tune]
        grid_f = [float(x) for x in args.grid_frame.split(",")]
        grid_o = [float(x) for x in args.grid_onset.split(",")] if has_onset else [args.onset_thr]
        best = None
        for ft, ot in itertools.product(grid_f, grid_o):
            s = score_predictions(tune, tune_preds, frame_thr=ft, onset_thr=ot, **common)
            f1 = s["note"][args.tune_metric]
            print(f"  tune frame_thr={ft} onset_thr={ot}: {args.tune_metric} {f1:.3f}")
            tuning.append({"frame_thr": ft, "onset_thr": ot, args.tune_metric: f1})
            if best is None or f1 > best[0]:
                best = (f1, ft, ot)
        _, frame_thr, onset_thr = best
        print(f"chosen on {Path(args.tune_dir).name} by {args.tune_metric}: "
              f"frame_thr={frame_thr} onset_thr={onset_thr}")
        at_edge = frame_thr in (min(grid_f), max(grid_f)) or (
            has_onset and onset_thr in (min(grid_o), max(grid_o)))
        if at_edge:
            print("  note: a chosen threshold is at the edge of the grid; widen it if you can")

    pieces = load_pieces(args.dir, args.max_pieces)
    preds = [predict_piece(model, p.cqt, device=device) for p in pieces]
    result = score_predictions(pieces, preds, frame_thr=frame_thr, onset_thr=onset_thr,
                               per_piece=True, **common)
    result["ci"] = summarize(result["per_piece"])
    result.update({"checkpoint": args.ckpt, "epoch": ck_epoch, "frame_thr": frame_thr,
                   "onset_thr": onset_thr, "min_frames": args.min_frames, "onset_head": has_onset,
                   "dir": args.dir, "tune_metric": args.tune_metric, "tuning": tuning})

    f, n, ci = result["frame"], result["note"], result["ci"]
    span = lambda k: f"[{ci[k]['lo']:.3f}, {ci[k]['hi']:.3f}]"  # noqa: E731
    print(f"\n{result['n_pieces']} pieces (95% CI over pieces in brackets)")
    print(f"frame F1 {f['f1']:.3f} {span('frame_f1')} (P {f['precision']:.3f}, R {f['recall']:.3f})")
    print(f"note onset F1 {n['onset_f1']:.3f} {span('note_onset_f1')} "
          f"(P {n['onset_precision']:.3f}, R {n['onset_recall']:.3f})")
    print(f"note onset+offset F1 {n['onset_offset_f1']:.3f} {span('note_onset_offset_f1')}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
