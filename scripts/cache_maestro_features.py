"""
Cache CQT features + labels for MAESTRO v3.0.0 (run this on Kaggle).

Uses the official split column of maestro-v3.0.0.csv (train / validation /
test); pieces are never re-split, so there is no cross-split leakage.

Per piece, writes <out>/<split>/<stem>.npz containing:
  cqt          float16 (88, n_frames)   log-magnitude CQT
  frame_roll   uint8   (88, n_frames)   0/1 frame targets
  onset_roll   uint8   (88, n_frames)   0/1 onset targets
  ref_onsets, ref_offsets, ref_pitches  continuous, pedal-extended, deduped
                                        reference notes (for note-level eval)
  stem
and <out>/<split>_manifest.json listing the cached pieces.

Resume-safe: pieces whose .npz already exists are skipped, so a session that
dies halfway can simply be re-run.

Examples:
  # timing test on 6 pieces
  python scripts/cache_maestro_features.py --root $ROOT --out /kaggle/working/maestro --split validation --limit 6
  # ~40 h of train, deterministic random subset
  python scripts/cache_maestro_features.py --root $ROOT --out /kaggle/working/maestro --split train --max-hours 40
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from amt.config import Config
from amt.data.dataset import process_song
from amt.data.midi import dedupe_notes, extend_notes_with_sustain_pedal

CFG = Config()


def select_pieces(df: pd.DataFrame, split: str, max_hours: float | None,
                  limit: int | None, seed: int) -> pd.DataFrame:
    part = df[df["split"] == split].sample(frac=1.0, random_state=seed)  # deterministic shuffle
    if max_hours is not None:
        keep = part["duration"].cumsum() <= max_hours * 3600
        part = part[keep]
    if limit is not None:
        part = part.head(limit)
    return part


def cache_one(job: tuple[str, str, str]) -> dict:
    wav_path, midi_path, out_path = job
    stem = Path(wav_path).stem
    if os.path.exists(out_path):
        return {"stem": stem, "status": "skipped"}
    t0 = time.time()
    try:
        song = process_song(Path(wav_path), Path(midi_path), CFG)
        notes = dedupe_notes(extend_notes_with_sustain_pedal(
            midi_path, threshold=CFG.labels.sustain_pedal_threshold))
        tmp = out_path + ".tmp.npz"
        np.savez_compressed(
            tmp,
            cqt=song.cqt.astype(np.float16),
            frame_roll=song.frame_roll.astype(np.uint8),
            onset_roll=song.onset_roll.astype(np.uint8),
            ref_onsets=np.array([n.start for n in notes], dtype=np.float64),
            ref_offsets=np.array([n.end for n in notes], dtype=np.float64),
            ref_pitches=np.array([n.pitch for n in notes], dtype=np.int64),
            stem=stem,
        )
        os.replace(tmp, out_path)
        secs = song.cqt.shape[1] * CFG.cqt.hop_length / CFG.audio.sample_rate
        return {"stem": stem, "status": "ok", "audio_seconds": secs,
                "wall_seconds": time.time() - t0, "n_notes": len(notes)}
    except Exception as e:  # keep going; report at the end
        return {"stem": stem, "status": "FAILED", "error": repr(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing maestro-v3.0.0.csv and year folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True, choices=["train", "validation", "test"])
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap number of pieces (timing tests)")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    root, out_dir = Path(args.root), Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / "maestro-v3.0.0.csv")
    part = select_pieces(df, args.split, args.max_hours, args.limit, args.seed)
    hours = part["duration"].sum() / 3600
    print(f"{args.split}: {len(part)} pieces, {hours:.1f} h selected, {args.workers} workers")

    jobs = [
        (str(root / r.audio_filename), str(root / r.midi_filename),
         str(out_dir / f"{Path(r.audio_filename).stem}.npz"))
        for r in part.itertuples()
    ]

    t0 = time.time()
    results = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(cache_one, jobs), 1):
            results.append(res)
            if res["status"] == "FAILED":
                print(f"[{i}/{len(jobs)}] FAILED {res['stem']}: {res['error']}")
            elif i % 10 == 0 or i == len(jobs):
                done_audio = sum(r.get("audio_seconds", 0) for r in results)
                wall = time.time() - t0
                print(f"[{i}/{len(jobs)}] {done_audio/3600:.1f} h of audio in {wall/60:.1f} min "
                      f"({done_audio/max(wall,1e-9):.1f}x realtime)")

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "FAILED"]
    manifest = {
        "split": args.split,
        "stems": sorted(Path(j[2]).stem for j in jobs if os.path.exists(j[2])),
        "seed": args.seed, "max_hours": args.max_hours,
    }
    (Path(args.out) / f"{args.split}_manifest.json").write_text(json.dumps(manifest, indent=1))
    size_gb = sum(f.stat().st_size for f in out_dir.glob("*.npz")) / 1e9
    print(f"\n{len(ok)} ok, {len(results)-len(ok)-len(failed)} skipped, {len(failed)} failed; "
          f"{args.split} cache is {size_gb:.2f} GB in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()