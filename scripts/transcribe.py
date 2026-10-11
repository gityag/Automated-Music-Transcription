"""
Transcribe a piano recording (audio file) to a MIDI file.

Usage:
    python scripts/transcribe.py recording.wav --ckpt run2/best.pt --out recording.mid

    # only a slice (seconds), handy for long files:
    python scripts/transcribe.py recording.mp3 --ckpt run2/best.pt --offset 30 --duration 20

Thresholds default to the values tuned on MAESTRO validation; override
with --frame-thr / --onset-thr. Needs torch, librosa and pretty_midi.
Any format librosa/soundfile/ffmpeg can read works (wav, flac, mp3, ...).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from amt.inference import notes_to_midi, transcribe_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="input audio file")
    ap.add_argument("--ckpt", required=True, help="checkpoint, e.g. run2/best.pt")
    ap.add_argument("--out", default=None, help="output .mid (default: next to the audio file)")
    ap.add_argument("--offset", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--duration", type=float, default=None, help="seconds to transcribe")
    ap.add_argument("--frame-thr", type=float, default=None)
    ap.add_argument("--onset-thr", type=float, default=None)
    ap.add_argument("--min-frames", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(args.audio).with_suffix(".mid")
    notes = transcribe_file(
        args.audio, args.ckpt, offset=args.offset, duration=args.duration,
        frame_thr=args.frame_thr, onset_thr=args.onset_thr, min_frames=args.min_frames,
    )
    notes_to_midi(notes, out)

    if notes:
        span = max(n.end for n in notes)
        lo, hi = min(n.pitch for n in notes), max(n.pitch for n in notes)
        print(f"{len(notes)} notes, {span:.1f} s, pitch range {lo}-{hi}")
    else:
        print("no notes detected")
    print(f"written to {out}")


if __name__ == "__main__":
    main()