from __future__ import annotations

import pytest

from amt.config import Config, CQTConfig, LabelConfig


def test_defaults_are_internally_consistent():
    Config()  # must not raise


def test_yaml_roundtrip(tmp_path):
    cfg = Config()
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    cfg2 = Config.from_yaml(path)
    assert cfg == cfg2


def test_partial_yaml_overrides_only_given_fields(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  epochs: 5\n  batch_size: 8\n")
    cfg = Config.from_yaml(path)
    assert cfg.train.epochs == 5
    assert cfg.train.batch_size == 8
    assert cfg.train.learning_rate == Config().train.learning_rate  # untouched default


def test_mismatched_fmin_and_pitch_low_raises():
    with pytest.raises(ValueError, match="fmin_midi"):
        Config(cqt=CQTConfig(fmin_midi=20))


def test_mismatched_n_bins_and_pitch_range_raises():
    with pytest.raises(ValueError, match="n_bins"):
        Config(labels=LabelConfig(pitch_low=21, pitch_high=100))  # 80 notes != 88 bins
