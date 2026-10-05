"""
Score a trained checkpoint with frame- and note-level metrics.

Optionally tunes the decoding thresholds on a separate (validation)
folder first, then applies the chosen thresholds to the evaluation
folder -- never tune on the data you report.

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
import dataclasses
import itertools
import json
import random
from pathlib import Path

import torch

from amt.config import Config, _resolve
from amt.data.cache import load_piece
from amt.evaluation.decode import score_predictions
from amt.models.cnn import build_model, predict_piece
from amt.training.loop import pick_device


def config_from_dict(d: dict) -> Config:
    kwargs = {f.name: _resolve(f.type, d[f.name]) for f in dataclasses.fields(Config) if f.name in d}
    return Config(**kwargs)


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
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ck["config"])
    device = pick_device(cfg.train.device)
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    fs = cfg.audio.sample_rate / cfg.cqt.hop_length
    common = dict(fs=fs, pitch_low=cfg.labels.pitch_low, min_frames=args.min_frames)
    has_onset = model.onset_out is not None
    print(f"checkpoint epoch {ck.get('epoch')}, onset head: {has_onset}, device: {device}")

    frame_thr, onset_thr = args.frame_thr, args.onset_thr
    if args.tune_dir:
        tune = load_pieces(args.tune_dir, args.tune_pieces)
        tune_preds = [predict_piece(model, p.cqt, device=device) for p in tune]
        grid_f = [0.3, 0.4, 0.5, 0.6, 0.7]
        grid_o = [0.3, 0.4, 0.5, 0.6, 0.7] if has_onset else [args.onset_thr]
        best = None
        for ft, ot in itertools.product(grid_f, grid_o):
            s = score_predictions(tune, tune_preds, frame_thr=ft, onset_thr=ot, **common)
            f1 = s["note"]["onset_f1"]
            print(f"  tune frame_thr={ft} onset_thr={ot}: note onset F1 {f1:.3f}")
            if best is None or f1 > best[0]:
                best = (f1, ft, ot)
        _, frame_thr, onset_thr = best
        print(f"chosen on {Path(args.tune_dir).name}: frame_thr={frame_thr} onset_thr={onset_thr}")

    pieces = load_pieces(args.dir, args.max_pieces)
    preds = [predict_piece(model, p.cqt, device=device) for p in pieces]
    result = score_predictions(pieces, preds, frame_thr=frame_thr, onset_thr=onset_thr, **common)
    result.update({"checkpoint": args.ckpt, "epoch": ck.get("epoch"), "frame_thr": frame_thr,
                   "onset_thr": onset_thr, "min_frames": args.min_frames, "onset_head": has_onset,
                   "dir": args.dir})

    f, n = result["frame"], result["note"]
    print(f"\n{result['n_pieces']} pieces")
    print(f"frame F1 {f['f1']:.3f} (P {f['precision']:.3f}, R {f['recall']:.3f})")
    print(f"note onset F1 {n['onset_f1']:.3f} (P {n['onset_precision']:.3f}, R {n['onset_recall']:.3f})")
    print(f"note onset+offset F1 {n['onset_offset_f1']:.3f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()