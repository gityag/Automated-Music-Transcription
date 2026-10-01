"""
Train the frame CNN on cached MAESTRO features.

Examples:
  # quick smoke test (10 pieces, 2 short epochs)
  python scripts/train.py --cache /kaggle/working/maestro --out /kaggle/working/run0 \
      --max-train-hours 2 --max-val-pieces 6 --epochs 2 --steps-per-epoch 50

  # full run with the config's settings
  python scripts/train.py --cache /kaggle/working/maestro --out /kaggle/working/run1 --config configs/default.yaml
"""
from __future__ import annotations

import argparse

from amt.config import Config
from amt.data.cache import load_split, total_hours
from amt.training.loop import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="folder with train/ validation/ test/ subfolders")
    ap.add_argument("--out", required=True, help="where checkpoints and history.json go")
    ap.add_argument("--config", default=None, help="YAML config (defaults if omitted)")
    ap.add_argument("--max-train-hours", type=float, default=None)
    ap.add_argument("--max-val-pieces", type=int, default=24, help="validation pieces scored each epoch")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    fs = cfg.audio.sample_rate / cfg.cqt.hop_length

    train_pieces = load_split(args.cache, "train", max_hours=args.max_train_hours, seed=cfg.train.seed, fs=fs)
    val_pieces = load_split(args.cache, "validation", max_pieces=args.max_val_pieces, seed=cfg.train.seed, fs=fs)
    print(f"loaded {len(train_pieces)} train pieces ({total_hours(train_pieces, fs):.1f} h), "
          f"{len(val_pieces)} val pieces ({total_hours(val_pieces, fs):.1f} h)")

    train(cfg, train_pieces, val_pieces, args.out, epochs=args.epochs,
          steps_per_epoch=args.steps_per_epoch, resume=not args.no_resume)


if __name__ == "__main__":
    main()
