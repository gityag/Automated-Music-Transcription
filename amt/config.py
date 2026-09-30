"""
Central configuration for the AMT pipeline.

Every constant that the legacy scripts hardcoded or silently overrode
(sample rate, pitch range, frame rate, seed, ...) lives here instead,
as one typed, serializable object. A run is fully specified by a
Config plus a Git commit hash -- that pair is what we log alongside
every result so experiments stay reproducible.

Usage:
    cfg = Config()                          # defaults
    cfg = Config.from_yaml("configs/default.yaml")
    cfg.to_yaml("configs/my_run.yaml")
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class AudioConfig:
    # A piano's highest fundamental is ~4.2 kHz (C8); 16 kHz sample rate
    # gives headroom to ~8 kHz (Nyquist) at 1/4 the samples of the legacy
    # 88.2 kHz load, with no loss of musically relevant content.
    sample_rate: int = 16_000


@dataclass
class CQTConfig:
    # librosa's CQT below is anchored at fmin; pitch_low/pitch_high in
    # LabelConfig MUST match this range or the input and target pitch
    # axes silently disagree (a bug present in the original pipeline).
    fmin_midi: int = 21          # A0 -- lowest note on an 88-key piano
    n_bins: int = 88             # A0..C8 inclusive, one bin per semitone
    bins_per_octave: int = 12
    # Frame rate shared with the piano-roll target. 16 ms hop => a
    # ~32 ms analysis frame at bins_per_octave=12, comfortably inside
    # mir_eval's default 50 ms onset tolerance (the legacy 83 ms grid
    # was not).
    hop_length: int = 256        # at sample_rate=16000 -> 16 ms/frame


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
    # Split BY SONG, never by segment: adjacent segments overlap in
    # audio, so a random segment-level split leaks test audio into
    # training (the bug that made the original accuracy numbers
    # meaningless).
    train_hours: float = 45.0    # subset of MAESTRO train split we actually use
    val_hours: float = 5.0
    seed: int = 21


@dataclass
class ModelConfig:
    name: str = "cnn_baseline"
    # A list, not a tuple: YAML has no tuple type, so a value that
    # round-trips through to_yaml/from_yaml would silently become a
    # list anyway. Keeping the Python type and the YAML type the same
    # avoids that mismatch.
    conv_channels: list[int] = field(default_factory=lambda: [32, 32, 64, 64])
    dropout: float = 0.25
    onset_head: bool = False     # ablation switch: frame-only vs onset+frame


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    seed: int = 21
    device: str = "auto"         # "auto" resolves to cuda > mps > cpu at runtime


@dataclass
class EvalConfig:
    onset_tolerance_seconds: float = 0.05
    offset_ratio: float = 0.2    # mir_eval default: max(50ms, 0.2 * note duration)


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
                f"labels.pitch_low ({self.labels.pitch_low}) or the CQT "
                "input and the piano-roll target cover different pitches."
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
    # dataclasses.fields(cls) gives string type names when annotations
    # use `from __future__ import annotations`; resolve via globals().
    if isinstance(section_cls, str):
        section_cls = globals()[section_cls]
    return section_cls(**values)
