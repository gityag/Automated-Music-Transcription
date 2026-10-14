"""
Reference row: Kong et al. (2021)'s released piano transcription model,
scored with OUR metric code on SMD.

SMD only, on purpose: Kong's checkpoint was trained on MAESTRO, and SMD was
never in its training data, so there is no train/test overlap to worry
about. For MAESTRO we cite Kong's published test numbers instead.

Needs: pip install piano_transcription_inference (downloads a ~165 MB
checkpoint from Zenodo on first use, so internet must be on), the SMD
audio, and our SMD cache (for reference notes and frame counts).

The output JSON has the same layout as scripts/evaluate_model.py
(frame, note, ci, per_piece), so scripts/compare_runs.py can compare it
piece by piece with any of our models on SMD.

Usage (Kaggle, GPU):
  python scripts/eval_kong_smd.py --audio-dir $SMD_WAV --cache-dir $SMD \
      --out /kaggle/working/model_smd_kong_ci.json
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


def macro(rows: list[dict], group: str) -> dict:
    return {k: float(np.mean([r[group][k] for r in rows])) for k in rows[0][group]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True, help="folder with the SMD .wav files")
    ap.add_argument("--cache-dir", required=True, help="folder with our SMD .npz files")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import librosa
    import torch
    from piano_transcription_inference import PianoTranscription, sample_rate

    cfg = Config()
    fs = cfg.audio.sample_rate / cfg.cqt.hop_length
    model = PianoTranscription(device=torch.device(args.device))
    wavs = {p.stem: p for p in Path(args.audio_dir).rglob("*.wav")}
    npzs = sorted(Path(args.cache_dir).rglob("*.npz"))
    missing = [p.stem for p in npzs if p.stem not in wavs]
    if missing:
        raise SystemExit(f"no audio for {len(missing)} cached pieces, e.g. {missing[:3]}")

    rows = []
    for i, npz in enumerate(npzs, 1):
        piece = load_piece(npz, with_notes=True)
        audio, _ = librosa.load(wavs[piece.stem], sr=sample_rate, mono=True)
        events = model.transcribe(audio, None)["est_note_events"]
        rows.append(score_external_notes(events_to_notes(events), piece, fs=fs,
                                         pitch_low=cfg.labels.pitch_low,
                                         pitch_high=cfg.labels.pitch_high))
        print(f"[{i}/{len(npzs)}] {piece.stem}: note onset F1 {rows[-1]['note']['onset_f1']:.3f}", flush=True)

    result = {"n_pieces": len(rows), "frame": macro(rows, "frame"), "note": macro(rows, "note"),
              "per_piece": rows, "system": "Kong et al. 2021 (piano_transcription_inference, default thresholds)"}
    result["ci"] = summarize(rows)
    f, n, ci = result["frame"], result["note"], result["ci"]
    span = lambda k: f"[{ci[k]['lo']:.3f}, {ci[k]['hi']:.3f}]"  # noqa: E731
    print(f"\n{len(rows)} pieces (95% CI over pieces in brackets)")
    print(f"frame F1 {f['f1']:.3f} {span('frame_f1')}")
    print(f"note onset F1 {n['onset_f1']:.3f} {span('note_onset_f1')}")
    print(f"note onset+offset F1 {n['onset_offset_f1']:.3f} {span('note_onset_offset_f1')}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
