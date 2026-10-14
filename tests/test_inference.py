"""
Plumbing tests for amt.inference: audio file -> checkpoint -> notes -> MIDI.

These use a freshly initialised (untrained) model, so they check that the
pieces fit together and the MIDI file is valid, not that the transcription
is any good. Accuracy is measured by scripts/evaluate_model.py.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")

from amt.config import Config
from amt.data.midi import Note, load_notes
from amt.inference import (
    DEFAULT_THRESHOLDS,
    audio_to_cqt,
    load_checkpoint,
    notes_to_midi,
    transcribe_file,
)
from amt.models.cnn import build_model


def _make_ckpt(path, onset_head):
    cfg = Config()
    cfg.model.onset_head = onset_head
    cfg.model.conv_channels = [4, 4, 4, 4]   # tiny: keep the test fast
    model = build_model(cfg)
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg), "epoch": 1}, path)
    return cfg


def _make_wav(path, seconds=1.5, sr=16000):
    t = np.arange(int(seconds * sr)) / sr
    audio = 0.3 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    sf.write(path, audio, sr)


def test_notes_to_midi_round_trip(tmp_path):
    notes = [Note(60, 0.5, 1.0, 100), Note(64, 0.5, 1.5, 100), Note(60, 1.2, 1.8, 100)]
    out = tmp_path / "x.mid"
    notes_to_midi(notes, out)
    back = load_notes(str(out))
    assert [(n.pitch, round(n.start, 3), round(n.end, 3)) for n in back] == \
           [(60, 0.5, 1.0), (64, 0.5, 1.5), (60, 1.2, 1.8)]


def test_audio_to_cqt_shape_and_dtype():
    cfg = Config()
    audio = np.zeros(16000, dtype=np.float32)
    cqt = audio_to_cqt(audio, 16000, cfg)
    assert cqt.shape[0] == cfg.cqt.n_bins
    assert cqt.dtype == np.float32
    assert cqt.shape[1] == 1 + 16000 // cfg.cqt.hop_length


@pytest.mark.parametrize("onset_head", [True, False])
def test_transcribe_file_end_to_end(tmp_path, onset_head):
    ckpt, wav, mid = tmp_path / "m.pt", tmp_path / "a.wav", tmp_path / "a.mid"
    _make_ckpt(ckpt, onset_head)
    _make_wav(wav)
    model, cfg, _ = load_checkpoint(ckpt, device="cpu")
    assert (model.onset_out is not None) == onset_head

    notes = transcribe_file(wav, ckpt)
    assert all(isinstance(n, Note) for n in notes)
    assert all(21 <= n.pitch <= 108 and n.end > n.start >= 0 for n in notes)

    notes_to_midi(notes, mid)
    assert len(load_notes(str(mid))) == len(notes)


def test_default_thresholds_cover_every_released_model_type():
    from amt.inference import default_thresholds
    from amt.models.cnn import FrameCNN
    assert default_thresholds(FrameCNN(onset_head=False)) == {"frame_thr": 0.5, "onset_thr": 0.5}
    assert default_thresholds(FrameCNN(onset_head=True)) == {"frame_thr": 0.3, "onset_thr": 0.8}
    assert default_thresholds(FrameCNN(onset_head=True, temporal="bigru")) == {"frame_thr": 0.5, "onset_thr": 0.9}
    for d in DEFAULT_THRESHOLDS.values():
        assert 0 < d["frame_thr"] < 1 and 0 < d["onset_thr"] < 1
