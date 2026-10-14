"""
Robustness of our model vs Kong et al.'s under controlled degradations of
the SMD audio: white noise, synthetic reverb, a 4 kHz low-pass and
detuning (amt.evaluation.degrade). Both systems see exactly the same
degraded audio and are scored with the same metric code against the same
reference notes (time-rescaled for the detuning conditions).

Frame F1 here is computed from each system's NOTES rasterised on our frame
grid (the only way to score Kong's note output at frame level), so it is
not the same number as evaluate_model.py's frame F1; compare systems
within this table only.

One JSON per (system, condition) is written as soon as it finishes, so a
run can be split across sessions (--conditions / --systems) and resumed.

Usage (Kaggle, GPU, internet on for the Kong checkpoint):
  python scripts/robustness.py --audio-dir $WAV --cache-dir $SMD --ckpt run4_best.pt \
      --out-dir /kaggle/working/robustness
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np

from amt.data.cache import load_piece
from amt.data.midi import Note, notes_to_piano_roll
from amt.evaluation.degrade import CONDITIONS
from amt.evaluation.external import events_to_notes, score_external_notes
from amt.evaluation.stats import summarize
from amt.inference import load_checkpoint, transcribe_audio


def scaled_piece(piece, scale: float, n_frames: int, fs: float):
    ref = [Note(int(p), float(s) * scale, float(e) * scale, 100)
           for p, s, e in zip(piece.ref_pitches, piece.ref_onsets, piece.ref_offsets)]
    frame, _ = notes_to_piano_roll(ref, n_frames=n_frames, fs=fs, pitch_low=21, pitch_high=108)
    return SimpleNamespace(stem=piece.stem, frame=frame.astype(np.uint8),
                           ref_pitches=np.array([n.pitch for n in ref]),
                           ref_onsets=np.array([n.start for n in ref]),
                           ref_offsets=np.array([n.end for n in ref]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ckpt", required=True, help="our checkpoint (e.g. run4 best.pt)")
    ap.add_argument("--systems", default="ours,kong")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--max-pieces", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    systems = args.systems.split(",")
    conditions = args.conditions.split(",")
    model, cfg, device = load_checkpoint(args.ckpt)
    sr, hop = cfg.audio.sample_rate, cfg.cqt.hop_length
    fs = sr / hop
    kong = None
    if "kong" in systems:
        import torch
        from piano_transcription_inference import PianoTranscription
        kong = PianoTranscription(device=torch.device(args.device))

    wavs = {p.stem: p for p in Path(args.audio_dir).rglob("*.wav")}
    npzs = sorted(Path(args.cache_dir).rglob("*.npz"))[: args.max_pieces]
    pieces = [load_piece(p, with_notes=True) for p in npzs]
    audio = {pc.stem: librosa.load(wavs[pc.stem], sr=sr, mono=True)[0] for pc in pieces}

    for cond in conditions:
        rows = {s: [] for s in systems}
        todo = [s for s in systems if not (out_dir / f"{s}_{cond}.json").exists()]
        if not todo:
            print(f"{cond}: done already, skipping")
            continue
        for i, pc in enumerate(pieces, 1):
            x, scale = CONDITIONS[cond](audio[pc.stem], sr)
            ref = scaled_piece(pc, scale, n_frames=1 + len(x) // hop, fs=fs)
            if "ours" in todo:
                est = transcribe_audio(x, sr, model, cfg, device=device)
                rows["ours"].append(score_external_notes(est, ref, fs=fs))
            if "kong" in todo:
                est = events_to_notes(kong.transcribe(x.astype(np.float64), None)["est_note_events"])
                rows["kong"].append(score_external_notes(est, ref, fs=fs))
            print(f"  {cond} [{i}/{len(pieces)}] {pc.stem}", flush=True)
        for s in todo:
            r = rows[s]
            res = {"system": s, "condition": cond, "n_pieces": len(r),
                   "frame": {k: float(np.mean([x["frame"][k] for x in r])) for k in r[0]["frame"]},
                   "note": {k: float(np.mean([x["note"][k] for x in r])) for k in r[0]["note"]},
                   "ci": summarize(r), "per_piece": r}
            (out_dir / f"{s}_{cond}.json").write_text(json.dumps(res, indent=1))
            print(f"{cond:14s} {s:5s} frame {res['frame']['f1']:.3f}  note {res['note']['onset_f1']:.3f}  "
                  f"on+off {res['note']['onset_offset_f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
