from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from amt.config import Config
from amt.data.midi import extend_notes_with_sustain_pedal, notes_to_piano_roll
from amt.features.cqt import compute_log_cqt, frame_rate

@dataclass(frozen=True)
class SongFeatures:
    stem: str
    cqt: np.ndarray
    frame_roll: np.ndarray
    onset_roll: np.ndarray

def process_song(wav_path: Path, midi_path: Path, cfg: Config) -> SongFeatures:
    """
    Load one song's audio and MIDI, compute its CQT and piano-roll labels.
    """
    audio, sr = librosa.load(wav_path, sr=cfg.audio.sample_rate)
    fs = frame_rate(sr, cfg.cqt.hop_length)
    cqt = compute_log_cqt(audio, sr, cfg.cqt.fmin_midi, cfg.cqt.n_bins, cfg.cqt.bins_per_octave, cfg.cqt.hop_length)
    notes = extend_notes_with_sustain_pedal(midi_path, threshold=cfg.labels.sustain_pedal_threshold)
    frame_roll, onset_roll = notes_to_piano_roll(notes, n_frames=cqt.shape[1], fs=fs, pitch_low=cfg.labels.pitch_low, pitch_high=cfg.labels.pitch_high)
    return SongFeatures(stem=wav_path.stem, cqt=cqt, frame_roll=frame_roll, onset_roll=onset_roll)