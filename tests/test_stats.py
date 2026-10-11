"""Tests for amt.evaluation.stats (pure NumPy)."""
from __future__ import annotations

import numpy as np
import pytest

from amt.evaluation.stats import bootstrap_ci, paired_bootstrap, summarize


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    x = np.random.default_rng(1).uniform(0.6, 0.9, size=177)
    r = bootstrap_ci(x)
    assert r["lo"] < r["mean"] < r["hi"]
    assert r["mean"] == pytest.approx(x.mean())
    assert bootstrap_ci(x) == r                      # fixed seed -> same interval
    assert r["hi"] - r["lo"] < 0.05                 # 177 pieces: a tight interval


def test_bootstrap_ci_of_constant_is_a_point():
    r = bootstrap_ci([0.8] * 20)
    assert r["lo"] == pytest.approx(0.8) and r["hi"] == pytest.approx(0.8)


def test_paired_bootstrap_detects_consistent_small_gain():
    rng = np.random.default_rng(2)
    a = rng.uniform(0.5, 0.9, size=50)           # large spread across pieces...
    b = a + 0.01 + rng.normal(0, 0.003, size=50)  # ...but B is 1 point better on every piece
    r = paired_bootstrap(a, b)
    assert r["lo"] > 0 and r["p_le_zero"] < 0.01
    # the same gain is invisible to two separate intervals, which overlap
    ca, cb = bootstrap_ci(a), bootstrap_ci(b)
    assert cb["lo"] < ca["hi"]


def test_paired_bootstrap_identical_systems_is_zero():
    a = np.linspace(0.5, 0.9, 30)
    r = paired_bootstrap(a, a)
    assert r["mean_diff"] == 0 and r["lo"] == 0 and r["hi"] == 0


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])


def test_summarize_reads_score_predictions_per_piece_layout():
    per_piece = [{"stem": str(i), "frame": {"f1": 0.7 + 0.01 * i},
                  "note": {"onset_f1": 0.8, "onset_offset_f1": 0.5}} for i in range(10)]
    s = summarize(per_piece)
    assert set(s) == {"frame_f1", "note_onset_f1", "note_onset_offset_f1"}
    assert s["note_onset_f1"]["mean"] == pytest.approx(0.8)
