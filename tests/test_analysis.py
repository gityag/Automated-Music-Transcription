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


# ------------------------------------------------------- piece-level bootstrap

from amt.evaluation.analysis import (bootstrap_breakdown, bootstrap_group_gap,  # noqa: E402
                                     bootstrap_paired_diff, resample_weights)


def _toy(seed=0, n_pieces=30, per_piece=40):
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n_pieces), per_piece)
    dur = rng.uniform(0, 3, size=len(pid))
    # long notes are harder; each piece has its own difficulty
    p_hit = np.clip(0.9 - 0.15 * dur + rng.normal(0, 0.1, n_pieces)[pid], 0, 1)
    hit = rng.random(len(pid)) < p_hit
    return pid, dur, hit


def test_resample_weights_draw_each_resample_once_per_piece():
    w = resample_weights(7, n_resamples=50, seed=1)
    assert w.shape == (50, 7) and np.all(w.sum(axis=1) == 7)


def test_bootstrap_breakdown_matches_point_breakdown_and_brackets_it():
    pid, dur, hit = _toy()
    bins = [0, 1, 2, np.inf]
    plain = breakdown(dur, {"r": hit}, bins)
    boot = bootstrap_breakdown(dur, pid, {"r": hit}, bins, n_resamples=500)
    for a, b in zip(plain, boot):
        assert a["n"] == b["n"] and b["r"] == pytest.approx(a["r"])
        assert b["r_lo"] < b["r"] < b["r_hi"]


def test_bootstrap_breakdown_is_wider_when_pieces_disagree():
    # same pooled recall (0.5), but in one case whole pieces hit or miss together
    pid = np.repeat(np.arange(20), 10)
    vals = np.zeros(len(pid))
    mixed = np.tile([True, False], 100)
    clustered = pid % 2 == 0
    w_mixed = bootstrap_breakdown(vals, pid, {"r": mixed}, [0, 1], n_resamples=500)[0]
    w_clust = bootstrap_breakdown(vals, pid, {"r": clustered}, [0, 1], n_resamples=500)[0]
    assert w_mixed["r"] == w_clust["r"] == pytest.approx(0.5)
    assert w_mixed["r_hi"] - w_mixed["r_lo"] == pytest.approx(0.0)
    assert w_clust["r_hi"] - w_clust["r_lo"] > 0.2


def test_bootstrap_breakdown_respects_mask_and_empty_bins():
    pid, dur, hit = _toy()
    mask = dur < 1.0
    rows = bootstrap_breakdown(dur, pid, {"r": hit}, [0, 1, 2, np.inf], mask=mask, n_resamples=200)
    assert rows[0]["n"] == int(mask.sum())
    assert rows[1]["n"] == 0 and rows[1]["r"] is None and rows[1]["r_lo"] is None


def test_paired_diff_is_zero_for_identical_systems_and_positive_for_a_better_one():
    pid, dur, hit = _toy()
    bins = [0, 1, 2, np.inf]
    same = bootstrap_paired_diff(dur, pid, hit, hit, bins, n_resamples=300)
    assert all(r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0 for r in same)
    better = hit | (np.random.default_rng(3).random(len(hit)) < 0.5)
    rows = bootstrap_paired_diff(dur, pid, hit, better, bins, n_resamples=300)
    assert all(r["diff"] > 0 and r["lo"] > 0 and r["p_le_zero"] == 0 for r in rows)


def test_group_gap_sign_and_counts():
    pid, dur, _ = _toy()
    group = np.arange(len(pid)) % 3 == 0          # "pedal-extended"
    hit = ~group                                  # group notes always missed, others always hit
    rows = bootstrap_group_gap(dur, pid, hit, group, [0, 1, 2, np.inf], n_resamples=200)
    for r in rows:
        assert r["diff"] == pytest.approx(1.0) and r["n"] == r["n_group"] + r["n_rest"]
