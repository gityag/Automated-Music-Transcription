"""Central configuration for the AMT pipeline."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class AudioConfig:
    sample_rate: int = 16_000


@dataclass
class CQTConfig:
    fmin_midi: int = 21
    n_bins: int = 88
    bins_per_octave: int = 12
    hop_length: int = 256


@dataclass
class LabelConfig:
    pitch_low: int = 21          # A0 (MIDI number)
    pitch_high: int = 108        # C8 (MIDI number) -- 88 keys inclusive
    # Extend a note's sounding duration while the sustain pedal (CC64)
    # is held past its note-off, matching how the piano actually sounds
    # and how MAESTRO-based baselines (e.g. Onsets & Frames) define the
    # frame target. Must be applied identically at train and eval time.
    extend_with_sustain_pedal: bool = True
    sustain_pedal_threshold: int = 64  # CC64 value counted as "held"


@dataclass
class SplitConfig:
    train_hours: float = 45.0
    val_hours: float = 5.0
    seed: int = 21


@dataclass
class ModelConfig:
    name: str = "cnn_baseline"
    conv_channels: list[int] = field(default_factory=lambda: [32, 32, 64, 64])
    dropout: float = 0.25
    onset_head: bool = False
    # Temporal model after the per-frame layers: "none" (the CNN alone sees
    # +-15 frames) or "bigru" (a small bidirectional GRU over time, run on
    # each pitch separately with weights shared by all 88 pitches).
    temporal: str = "none"
    pitch_feat: int = 16      # features per pitch fed to the GRU
    rnn_hidden: int = 32      # GRU units per direction


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    seed: int = 21
    device: str = "auto"
    segment_frames: int = 256  # training window: 256 frames = ~4.1 s at 62.5 fps


@dataclass
class EvalConfig:
    onset_tolerance_seconds: float = 0.05
    offset_ratio: float = 0.2


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    cqt: CQTConfig = field(default_factory=CQTConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self) -> None:
        n_expected = self.labels.pitch_high - self.labels.pitch_low + 1
        if self.cqt.fmin_midi != self.labels.pitch_low:
            raise ValueError(
                f"cqt.fmin_midi ({self.cqt.fmin_midi}) must equal "
                f"labels.pitch_low ({self.labels.pitch_low})."
            )
        if self.cqt.n_bins != n_expected:
            raise ValueError(
                f"cqt.n_bins ({self.cqt.n_bins}) must equal "
                f"pitch_high - pitch_low + 1 ({n_expected})."
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        kwargs = {}
        for section_field in dataclasses.fields(cls):
            section_cls = section_field.type
            if section_field.name in raw:
                kwargs[section_field.name] = _resolve(section_cls, raw[section_field.name])
        return cls(**kwargs)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


def _resolve(section_cls, values: dict):
    if isinstance(section_cls, str):
        section_cls = globals()[section_cls]
    return section_cls(**values)
