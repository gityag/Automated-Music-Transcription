"""
Dataset assembly: per-song feature/label extraction, and song-level
(not segment-level) train/val/test splitting.

Splitting by song -- and more specifically by *group* (a group being a
whole musical work, so multi-movement pieces stay together) -- exists
because a random split at the segment level lets adjacent, overlapping
audio windows from the same recording leak across train and test. See
the Week 1 planning discussion and docs/decisions/ for the background
on why that leak makes reported accuracy numbers meaningless.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from amt.config import Config
from amt.data.midi import extend_notes_with_sustain_pedal, notes_to_piano_roll
from amt.features.cqt import compute_log_cqt, frame_rate


@dataclass(frozen=True)
class SongFeatures:
    """One song's extracted inputs and targets, bundled together."""
    stem: str
    cqt: np.ndarray
    frame_roll: np.ndarray
    onset_roll: np.ndarray


def process_song(wav_path: Path, midi_path: Path, cfg: Config) -> SongFeatures:
    """
    Load one song's audio and MIDI, compute its CQT and piano-roll labels.

    n_frames for the labels is taken from the CQT's own shape rather
    than recomputed from duration*fs, since those two ways of counting
    frames can silently disagree (see docs/decisions/0003 and the
    2026-09-25 PROGRESS.md entry for the bug this avoids).
    """
    audio, sr = librosa.load(wav_path, sr=cfg.audio.sample_rate)
    fs = frame_rate(sr, cfg.cqt.hop_length)
    cqt = compute_log_cqt(
        audio, sr, cfg.cqt.fmin_midi, cfg.cqt.n_bins,
        cfg.cqt.bins_per_octave, cfg.cqt.hop_length,
    )
    notes = extend_notes_with_sustain_pedal(
        midi_path, threshold=cfg.labels.sustain_pedal_threshold
    )
    frame_roll, onset_roll = notes_to_piano_roll(
        notes, n_frames=cqt.shape[1], fs=fs,
        pitch_low=cfg.labels.pitch_low, pitch_high=cfg.labels.pitch_high,
    )
    return SongFeatures(stem=wav_path.stem, cqt=cqt, frame_roll=frame_roll, onset_roll=onset_roll)


def group_durations(
    stem_to_duration: dict[str, float],
    stem_to_group: dict[str, str],
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """
    Aggregate per-stem durations into per-group totals, and record
    which stems belong to each group.
    """
    group_to_duration: dict[str, float] = {}
    group_to_stems: dict[str, list[str]] = {}
    for stem, duration in stem_to_duration.items():
        group = stem_to_group[stem]
        group_to_duration[group] = group_to_duration.get(group, 0) + duration
        group_to_stems.setdefault(group, []).append(stem)
    return group_to_duration, group_to_stems


def song_level_split(
    group_to_duration: dict[str, float],
    group_to_stems: dict[str, list[str]],
    cfg: Config,
) -> dict[str, list[str]]:
    """
    Randomly assign whole groups to train/val/test, greedily filling
    train to cfg.split.train_hours and val to cfg.split.val_hours
    (in seconds), with everything left over going to test.

    Shuffling uses a dedicated random.Random(cfg.split.seed) instance
    rather than the global random module, so this split's
    reproducibility can't be disturbed by unrelated code elsewhere
    calling random.* before this runs.
    """
    groups = list(group_to_duration.keys())
    rng = random.Random(cfg.split.seed)
    rng.shuffle(groups)

    train_seconds_target = cfg.split.train_hours * 3600
    val_seconds_target = cfg.split.val_hours * 3600

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    train_seconds = 0.0
    val_seconds = 0.0

    for group in groups:
        duration = group_to_duration[group]
        if train_seconds < train_seconds_target:
            splits["train"].extend(group_to_stems[group])
            train_seconds += duration
        elif val_seconds < val_seconds_target:
            splits["val"].extend(group_to_stems[group])
            val_seconds += duration
        else:
            splits["test"].extend(group_to_stems[group])

    return splits