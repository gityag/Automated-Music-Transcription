"""
Audio file in, notes (and a MIDI file) out.

Everything here mirrors how the training/evaluation data was made, so a
checkpoint sees the same kind of input it was trained on:
  - audio is loaded mono and resampled to cfg.audio.sample_rate,
  - the log-CQT uses the checkpoint's own CQT settings,
  - the CQT is rounded through float16 first, because the training
    caches store it as float16.
The checkpoint carries its config and the input normalization (as model
buffers), so nothing else has to be passed in.

Thresholds are decoding choices, not part of the model. The defaults are
the ones picked on MAESTRO validation by onset+offset F1
(see results/*_offs.json).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import torch

from amt.config import Config, _resolve
from amt.data.midi import Note
from amt.evaluation.decode import decode_notes
from amt.features.cqt import compute_log_cqt, frame_rate
from amt.models.cnn import build_model, predict_piece
from amt.training.loop import pick_device

# Tuned on MAESTRO validation (onset+offset F1).
DEFAULT_THRESHOLDS = {
    True: {"frame_thr": 0.3, "onset_thr": 0.8},    # model with an onset head (run2)
    False: {"frame_thr": 0.5, "onset_thr": 0.5},   # frame-only model (run1)
}


def config_from_dict(d: dict) -> Config:
    kwargs = {f.name: _resolve(f.type, d[f.name]) for f in dataclasses.fields(Config) if f.name in d}
    return Config(**kwargs)


def load_checkpoint(path: str | Path, device: str | None = None):
    """Returns (model, cfg, device). The model is in eval mode."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(ck["config"])
    dev = pick_device(device or cfg.train.device)
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    return model.to(dev).eval(), cfg, dev


def audio_to_cqt(audio: np.ndarray, sr: int, cfg: Config) -> np.ndarray:
    cqt = compute_log_cqt(audio, sr, cfg.cqt.fmin_midi, cfg.cqt.n_bins,
                          cfg.cqt.bins_per_octave, cfg.cqt.hop_length)
    return cqt.astype(np.float16).astype(np.float32)


def transcribe_audio(
    audio: np.ndarray,
    sr: int,
    model,
    cfg: Config,
    device=None,
    frame_thr: float | None = None,
    onset_thr: float | None = None,
    min_frames: int = 3,
) -> list[Note]:
    """audio must already be mono at sr == cfg.audio.sample_rate."""
    defaults = DEFAULT_THRESHOLDS[model.onset_out is not None]
    frame_thr = defaults["frame_thr"] if frame_thr is None else frame_thr
    onset_thr = defaults["onset_thr"] if onset_thr is None else onset_thr
    cqt = audio_to_cqt(audio, sr, cfg)
    frame_probs, onset_probs = predict_piece(model, cqt, device=device)
    return decode_notes(
        frame_probs, onset_probs,
        fs=frame_rate(sr, cfg.cqt.hop_length), pitch_low=cfg.labels.pitch_low,
        frame_thr=frame_thr, onset_thr=onset_thr, min_frames=min_frames,
    )


def transcribe_file(
    audio_path: str | Path,
    ckpt_path: str | Path,
    offset: float = 0.0,
    duration: float | None = None,
    **kwargs,
) -> list[Note]:
    model, cfg, device = load_checkpoint(ckpt_path)
    audio, sr = librosa.load(audio_path, sr=cfg.audio.sample_rate, mono=True,
                             offset=offset, duration=duration)
    return transcribe_audio(audio, sr, model, cfg, device=device, **kwargs)


def notes_to_midi(notes: list[Note], path: str | Path, tempo: float = 120.0) -> None:
    """Write notes (absolute seconds) as a single-piano MIDI file."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0, name="Piano")  # Acoustic Grand Piano
    for n in notes:
        piano.notes.append(pretty_midi.Note(velocity=int(n.velocity), pitch=int(n.pitch),
                                            start=float(n.start), end=float(n.end)))
    pm.instruments.append(piano)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))