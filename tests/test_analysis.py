"""Tests for amt.evaluation.analysis and amt.evaluation.degrade."""
from __future__ import annotations

import numpy as np
import pytest

from amt.data.midi import Note
from amt.evaluation.analysis import (breakdown, match_details, note_features,
                                     offset_error_summary, pedal_extended_mask)
from amt.evaluation.degrade import CONDITIONS, add_noise, detune


def test_match_details_labels_each_reference_note():
    ref = [Note(60, 0.0, 1.0, 100), Note(64, 0.5, 0.7, 100), Note(67, 2.0, 3.0, 100)]
    est = [Note(60, 0.02, 1.05, 100),   # onset + offset hit
           Note(64, 0.51, 1.50, 100),   # onset hit, ends 0.8 s too late
           Note(67, 2.30, 3.00, 100)]   # onset 300 ms late: miss
    d = match_details(ref, est)
    assert d["onset_hit"].tolist() == [True, True, False]
    assert d["oo_hit"].tolist() == [True, False, False]
    assert d["offset_err"][1] == pytest.approx(0.8)
    assert np.isnan(d["offset_err"][2])


def test_note_features_polyphony_counts_sounding_notes():
    ref = [Note(60, 0.0, 2.0, 100), Note(64, 0.5, 1.0, 100), Note(67, 0.5, 1.0, 100), Note(72, 3.0, 4.0, 100)]
    f = note_features(ref)
    assert f["polyphony"].tolist() == [1, 3, 3, 1]
    assert f["duration"].tolist() == pytest.approx([2.0, 0.5, 0.5, 1.0])


def test_pedal_extended_mask_compares_with_raw_midi_notes():
    raw = [Note(60, 0.0, 0.5, 100), Note(64, 1.0, 1.5, 100)]
    ref = [Note(60, 0.0, 1.2, 100), Note(64, 1.0, 1.5, 100)]   # first was extended by the pedal
    assert pedal_extended_mask(ref, raw).tolist() == [True, False]


def test_breakdown_and_offset_summary():
    vals = np.array([0.1, 0.3, 0.3, 3.0])
    rows = breakdown(vals, {"r": np.array([1, 0, 1, 1], dtype=bool)}, [0, 0.25, 1, np.inf])
    assert [r["n"] for r in rows] == [1, 2, 1]
    assert rows[1]["r"] == pytest.approx(0.5)
    s = offset_error_summary(np.array([-1.0, 0.0, 1.0, np.nan]), np.array([1.0, 1.0, 1.0, 1.0]))
    assert s["n_onset_matched"] == 3 and s["too_early"] == pytest.approx(1 / 3)


def test_noise_has_requested_snr_and_detune_shifts_pitch_and_time():
    sr = 16000
    t = np.arange(sr * 4) / sr
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y, scale = add_noise(x, sr, 10)
    snr = 10 * np.log10(np.mean(x ** 2) / np.mean((y - x) ** 2))
    assert scale == 1.0 and snr == pytest.approx(10, abs=0.1)
    y, scale = detune(x, sr, 50)
    peak = np.argmax(np.abs(np.fft.rfft(y[: sr * 2]))) * sr / (sr * 2)
    assert peak == pytest.approx(440 * 2 ** (50 / 1200), abs=1.0)
    assert scale == pytest.approx(2 ** (-50 / 1200))
    assert len(y) == pytest.approx(len(x) * scale, abs=2)


def test_every_condition_returns_audio_and_scale():
    x = np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32)
    for name, f in CONDITIONS.items():
        y, scale = f(x, 16000)
        assert y.dtype == np.float32 and np.isfinite(y).all(), name
        assert 0.9 < scale < 1.1, name


def test_robustness_reference_scaling_keeps_labels_aligned():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("rob", pathlib.Path(__file__).parents[1] / "scripts" / "robustness.py")
    rob = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rob)
    from types import SimpleNamespace
    from amt.evaluation.external import score_external_notes
    piece = SimpleNamespace(stem="p", ref_pitches=np.array([60, 64]),
                            ref_onsets=np.array([1.0, 2.0]), ref_offsets=np.array([1.5, 3.0]))
    sp = rob.scaled_piece(piece, 0.5, n_frames=200, fs=62.5)
    assert sp.ref_onsets.tolist() == [0.5, 1.0] and sp.ref_offsets.tolist() == [0.75, 1.5]
    est = [Note(60, 0.5, 0.75, 100), Note(64, 1.0, 1.5, 100)]   # notes at the sped-up times
    r = score_external_notes(est, sp)
    assert r["note"]["onset_offset_f1"] == pytest.approx(1.0) and r["frame"]["f1"] == pytest.approx(1.0)
