"""Plumbing test for the web demo (app.py): untrained tiny model, synthetic audio."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")
pytest.importorskip("matplotlib")

from amt.config import Config
from amt.data.midi import Note, load_notes
from amt.models.cnn import build_model


@pytest.fixture()
def app(tmp_path, monkeypatch):
    cfg = Config()
    cfg.model.onset_head, cfg.model.temporal, cfg.model.conv_channels = True, "bigru", [4, 4, 4, 4]
    ckpt = tmp_path / "m.pt"
    torch.save({"model": build_model(cfg).state_dict(), "config": dataclasses.asdict(cfg), "epoch": 1}, ckpt)
    monkeypatch.setenv("AMT_CKPT", str(ckpt))
    import app as module
    monkeypatch.setattr(module, "_MODEL", None)
    return module


def test_transcribe_returns_roll_midi_preview_and_summary(app, tmp_path):
    wav = tmp_path / "a.wav"
    t = np.arange(16000 * 2) / 16000
    sf.write(wav, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 16000)
    fig, midi, (sr, y), summary = app.transcribe(str(wav), 0.05, 0.05)   # low thresholds: some notes
    notes = load_notes(midi)
    assert len(notes) > 0 and f"{len(notes)} notes" in summary
    assert sr == app.PREVIEW_SR and y.dtype == np.float32 and np.max(np.abs(y)) <= 0.81
    assert fig.axes


def test_piano_roll_and_preview_handle_no_notes(app):
    assert app.piano_roll_figure([], 3.0).axes
    assert app.sine_preview([]).shape == (app.PREVIEW_SR,)
    assert app.sine_preview([Note(60, 0.0, 0.5, 100)]).size > 0


def test_transcribe_rejects_missing_audio(app):
    with pytest.raises(ValueError):
        app.transcribe(None, 0.5, 0.5)
