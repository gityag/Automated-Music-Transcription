"""
Reference row: Kong et al. (2021)'s released piano transcription model,
scored with OUR metric code and OUR reference notes, on any cached test set.

  - SMD: never in Kong's training data, so a clean out-of-domain comparison.
  - MAESTRO v3 test: Kong's checkpoint was trained on MAESTRO v2. The v2 and
    v3 test splits contain the same performances, so there is no train/test
    overlap; this run puts Kong and our models on identical pieces, labels
    (pedal-extended, deduped) and matching rules, instead of comparing with
    numbers published under a different protocol.

Needs: pip install piano_transcription_inference (downloads a ~165 MB
checkpoint from Zenodo on first use, so internet must be on), the audio,
and our feature cache for the same pieces (reference notes, frame counts).

Resume-safe: each scored piece is appended to <out>.partial.jsonl, and a
re-run skips pieces already there. The final JSON has the same layout as
scripts/evaluate_model.py (frame, note, ci, per_piece), so
scripts/compare_runs.py can compare it piece by piece with our models.

Usage (Kaggle, GPU):
  python scripts/eval_kong.py --audio-dir $SMD_WAV --cache-dir $SMD \
      --out results/model_smd_kong_ci.json
  python scripts/eval_kong.py --audio-dir $MAESTRO_ROOT --cache-dir $CACHE/test \
      --out results/model_maestro_test_kong_ci.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from amt.config import Config
from amt.data.cache import load_piece
from amt.evaluation.external import events_to_notes, score_external_notes
from amt.evaluation.stats import summarize

AUDIO_EXTS = (".wav", ".flac")


def macro(rows: list[dict], group: str) -> dict:
    return {k: float(np.mean([r[group][k] for r in rows])) for k in rows[0][group]}


def find_audio(root: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS:
            out.setdefault(p.stem, p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True, help="folder searched recursively for .wav/.flac")
    ap.add_argument("--cache-dir", required=True, help="folder with our .npz files for the same pieces")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N pieces (smoke test)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import librosa
    import torch
    from piano_transcription_inference import PianoTranscription, sample_rate

    cfg = Config()
    fs = cfg.audio.sample_rate / cfg.cqt.hop_length
    audio_files = find_audio(args.audio_dir)
    npzs = sorted(Path(args.cache_dir).rglob("*.npz"))
    if args.limit:
        npzs = npzs[: args.limit]
    if not npzs:
        raise SystemExit(f"no .npz files under {args.cache_dir}")
    missing = [p.stem for p in npzs if p.stem not in audio_files]
    if missing:
        raise SystemExit(f"no audio for {len(missing)} cached pieces, e.g. {missing[:3]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.jsonl")
    done: dict[str, dict] = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["stem"]] = row
        print(f"resuming: {len(done)} pieces already scored", flush=True)

    model = PianoTranscription(device=torch.device(args.device))
    rows = []
    for i, npz in enumerate(npzs, 1):
        if npz.stem in done:
            rows.append(done[npz.stem])
            continue
        piece = load_piece(npz, with_notes=True)
        audio, _ = librosa.load(audio_files[piece.stem], sr=sample_rate, mono=True)
        events = model.transcribe(audio, None)["est_note_events"]
        row = score_external_notes(events_to_notes(events), piece, fs=fs,
                                   pitch_low=cfg.labels.pitch_low,
                                   pitch_high=cfg.labels.pitch_high)
        row.setdefault("stem", piece.stem)
        rows.append(row)
        with partial.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[{i}/{len(npzs)}] {piece.stem}: note onset F1 {row['note']['onset_f1']:.3f}", flush=True)

    result = {"n_pieces": len(rows), "frame": macro(rows, "frame"), "note": macro(rows, "note"),
              "per_piece": rows, "system": "Kong et al. 2021 (piano_transcription_inference, default thresholds)"}
    result["ci"] = summarize(rows)
    f, n, ci = result["frame"], result["note"], result["ci"]
    span = lambda k: f"[{ci[k]['lo']:.3f}, {ci[k]['hi']:.3f}]"  # noqa: E731
    print(f"\n{len(rows)} pieces (95% CI over pieces in brackets)")
    print(f"frame F1 {f['f1']:.3f} {span('frame_f1')}")
    print(f"note onset F1 {n['onset_f1']:.3f} {span('note_onset_f1')}")
    print(f"note onset+offset F1 {n['onset_offset_f1']:.3f} {span('note_onset_offset_f1')}")
    out.write_text(json.dumps(result, indent=2))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
