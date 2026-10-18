"""
Web demo: upload a piano recording, get a piano roll, a MIDI file and a
quick sine-wave rendition of the transcription to compare by ear.

Runs on CPU. The checkpoint is found in this order:
  1. the AMT_CKPT environment variable (a local path),
  2. models/run4_best.pt,
  3. downloaded once from the GitHub release (AMT_CKPT_URL) into ~/.cache/amt/.

  pip install -e ".[train,demo]"
  python app.py                      # then open http://127.0.0.1:7860
"""
from __future__ import annotations

import os
import tempfile
import time
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
from matplotlib.figure import Figure

from amt.inference import default_thresholds, load_checkpoint, notes_to_midi, transcribe_audio

RELEASE_URL = "https://github.com/gityag/Automated-Music-Transcription/releases/download/v1.0/run4_best.pt"
MAX_SECONDS = float(os.environ.get("AMT_MAX_SECONDS", 600))   # keep CPU time bounded on a shared host
PREVIEW_SR = 16000

_MODEL = None


def checkpoint_path() -> Path:
    if os.environ.get("AMT_CKPT"):
        return Path(os.environ["AMT_CKPT"])
    local = Path(__file__).resolve().parent / "models" / "run4_best.pt"
    if local.exists():
        return local
    cached = Path.home() / ".cache" / "amt" / "run4_best.pt"
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        url = os.environ.get("AMT_CKPT_URL", RELEASE_URL)
        print(f"downloading checkpoint from {url}")
        tmp = cached.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(cached)
    return cached


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_checkpoint(checkpoint_path(), device="cpu")
    return _MODEL


def piano_roll_figure(notes, duration: float) -> Figure:
    fig = Figure(figsize=(10, 4), dpi=110)
    ax = fig.add_subplot()
    for n in notes:
        ax.barh(n.pitch, n.end - n.start, left=n.start, height=0.8, color="#2a78d6", linewidth=0)
    pitches = [n.pitch for n in notes] or [60]
    lo, hi = max(21, min(pitches) - 3), min(108, max(pitches) + 3)
    cs = [p for p in range(lo, hi + 1) if p % 12 == 0]
    ax.set_yticks(cs, [pretty_midi.note_number_to_name(p) for p in cs])
    ax.set_ylim(lo - 0.5, hi + 0.5)
    ax.set_xlim(0, max(duration, 1e-3))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e4e4e0", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def sine_preview(notes) -> np.ndarray:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes = [pretty_midi.Note(velocity=90, pitch=n.pitch, start=n.start, end=n.end) for n in notes]
    pm.instruments.append(inst)
    y = pm.synthesize(fs=PREVIEW_SR) if notes else np.zeros(PREVIEW_SR, dtype=np.float64)
    peak = np.max(np.abs(y)) or 1.0
    return (0.8 * y / peak).astype(np.float32)


def transcribe(audio_path, onset_thr: float, frame_thr: float):
    if not audio_path:
        raise ValueError("Upload or record some piano audio first.")
    model, cfg, device = get_model()
    sr = cfg.audio.sample_rate
    audio, _ = librosa.load(audio_path, sr=sr, mono=True, duration=MAX_SECONDS)
    duration = len(audio) / sr
    t0 = time.perf_counter()
    notes = transcribe_audio(audio, sr, model, cfg, device=device,
                             onset_thr=onset_thr, frame_thr=frame_thr)
    secs = time.perf_counter() - t0

    out_dir = Path(tempfile.mkdtemp(prefix="amt_"))
    midi_path = out_dir / (Path(audio_path).stem + "_transcribed.mid")
    notes_to_midi(notes, midi_path)

    clipped = " (only the first %d s were transcribed)" % MAX_SECONDS if duration >= MAX_SECONDS - 0.01 else ""
    summary = (f"{len(notes)} notes in {duration:.1f} s of audio{clipped}. "
               f"Transcribed in {secs:.2f} s on CPU ({duration / max(secs, 1e-6):.0f}x real time).")
    return piano_roll_figure(notes, duration), str(midi_path), (PREVIEW_SR, sine_preview(notes)), summary


def build_ui():
    import gradio as gr

    model, _, _ = get_model()
    thr = default_thresholds(model)
    with gr.Blocks(title="Piano transcription") as demo:
        gr.Markdown(
            "# Piano audio to MIDI\n"
            "A 3.8M-parameter CNN with a per-pitch GRU, trained on MAESTRO v3. Upload a solo piano "
            "recording (WAV, MP3, FLAC); you get a piano roll, a MIDI file and a plain sine-wave "
            "rendition of the notes to compare by ear. Works best on clean solo piano.")
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Piano recording")
                with gr.Accordion("Decoding thresholds", open=False):
                    onset = gr.Slider(0.05, 0.95, value=thr["onset_thr"], step=0.05,
                                      label="Onset threshold (higher = fewer, surer notes)")
                    frame = gr.Slider(0.05, 0.95, value=thr["frame_thr"], step=0.05,
                                      label="Frame threshold (higher = shorter notes)")
                run = gr.Button("Transcribe", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                roll = gr.Plot(label="Piano roll")
                with gr.Row():
                    midi = gr.File(label="MIDI file")
                    preview = gr.Audio(label="Transcription, sine-wave rendition", type="numpy")
        run.click(transcribe, [audio, onset, frame], [roll, midi, preview, summary])
    return demo


if __name__ == "__main__":
    build_ui().launch()
