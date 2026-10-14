"""
Model size and CPU speed: our checkpoint vs Kong et al.'s model, on the
same audio and the same machine. Reports parameters and the real-time
factor (processing seconds per second of audio; < 1 is faster than real
time), median of --repeats runs after one warm-up run.

Our time is split into feature extraction (CQT), the network, and note
decoding, so the paper can say where the time goes.

Usage (CPU; Kong's checkpoint must already be downloaded, see README):
  python scripts/benchmark_speed.py data/smd/wav_22050_mono/<piece>.wav \
      --ckpt models/run4_best.pt --seconds 60 --kong --out results/speed_cpu.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import librosa
import torch

from amt.evaluation.decode import decode_notes
from amt.inference import audio_to_cqt, default_thresholds, load_checkpoint
from amt.models.cnn import predict_piece


def timed(fn, repeats: int):
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")
    ap.add_argument("--kong", action="store_true", help="also time Kong et al.'s model")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    model, cfg, _ = load_checkpoint(args.ckpt, device="cpu")
    sr = cfg.audio.sample_rate
    x, _ = librosa.load(args.audio, sr=sr, mono=True, duration=args.seconds)
    dur = len(x) / sr
    thr = default_thresholds(model)
    fs = sr / cfg.cqt.hop_length

    t_cqt, cqt = timed(lambda: audio_to_cqt(x, sr, cfg), args.repeats)
    t_net, probs = timed(lambda: predict_piece(model, cqt, device="cpu"), args.repeats)
    t_dec, notes = timed(lambda: decode_notes(*probs, fs=fs, pitch_low=cfg.labels.pitch_low, **thr), args.repeats)
    ours_total = t_cqt + t_net + t_dec
    res = {"machine": f"{platform.machine()} {platform.processor() or platform.platform()}",
           "torch_threads": torch.get_num_threads(), "audio_seconds": dur,
           "ours": {"params": sum(p.numel() for p in model.parameters()),
                    "seconds": {"cqt": t_cqt, "network": t_net, "decode": t_dec, "total": ours_total},
                    "rtf": ours_total / dur, "n_notes": len(notes)}}
    print(f"audio {dur:.0f} s on {res['machine']}, {res['torch_threads']} threads")
    print(f"ours : {res['ours']['params']/1e6:.2f}M params, RTF {res['ours']['rtf']:.3f} "
          f"(CQT {t_cqt:.2f}s, network {t_net:.2f}s, decode {t_dec:.2f}s)")

    if args.kong:
        from piano_transcription_inference import PianoTranscription
        kong = PianoTranscription(device=torch.device("cpu"))
        t_k, _ = timed(lambda: kong.transcribe(x.astype("float64"), None), args.repeats)
        k_params = sum(p.numel() for p in kong.model.parameters())
        res["kong"] = {"params": k_params, "seconds": {"total": t_k}, "rtf": t_k / dur}
        print(f"kong : {k_params/1e6:.2f}M params, RTF {res['kong']['rtf']:.3f} ({t_k:.2f}s)")
        print(f"ours is {t_k / ours_total:.1f}x faster with {k_params / res['ours']['params']:.1f}x fewer parameters")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
