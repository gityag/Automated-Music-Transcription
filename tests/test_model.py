"""
Tests for the CNN, the segment sampler, and the training loop. Skipped
automatically where torch isn't installed (e.g. a laptop venv without the
`train` extra).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from amt.config import Config
from amt.data.cache import Piece, SegmentSampler, compute_norm_stats
from amt.models.cnn import FrameCNN, build_model, predict_piece
from amt.training.loop import evaluate_frames, train


def _synthetic_piece(n_frames=500, seed=0, stem="s"):
    """A trivially readable 'piece': an active pitch shows as a loud bin
    (plus a quieter octave harmonic) over a quiet noisy floor."""
    rng = np.random.default_rng(seed)
    cqt = rng.normal(-60, 2, size=(88, n_frames)).astype(np.float32)
    frame = np.zeros((88, n_frames), dtype=np.uint8)
    onset = np.zeros((88, n_frames), dtype=np.uint8)
    t = 0
    while t < n_frames - 40:
        pitch = int(rng.integers(0, 70))
        dur = int(rng.integers(20, 40))
        cqt[pitch, t:t + dur] = -10
        cqt[pitch + 12, t:t + dur] = -25
        frame[pitch, t:t + dur] = 1
        onset[pitch, t] = 1
        t += int(rng.integers(10, 30))
    return Piece(stem=stem, cqt=cqt.astype(np.float16), frame=frame, onset=onset)


def test_forward_shapes_and_onset_head():
    x = torch.randn(2, 88, 64)
    f, o = FrameCNN(onset_head=False)(x)
    assert f.shape == (2, 88, 64) and o is None
    f, o = FrameCNN(onset_head=True)(x)
    assert f.shape == (2, 88, 64) and o.shape == (2, 88, 64)


def test_build_model_follows_config():
    cfg = Config()
    m = build_model(cfg)
    assert m.n_bins == cfg.cqt.n_bins
    assert [c.out_channels for c in m.conv if isinstance(c, torch.nn.Conv2d)] == cfg.model.conv_channels


def test_chunked_prediction_matches_single_pass():
    # predict_piece pads each chunk with the receptive-field radius of real
    # context, so chunking must not change the output.
    torch.manual_seed(0)
    m = FrameCNN(conv_channels=(4, 4, 4, 4), hidden=16).eval()
    cqt = np.random.default_rng(1).normal(-40, 10, size=(88, 300)).astype(np.float32)
    full, _ = predict_piece(m, cqt, chunk=10_000)
    chunked, _ = predict_piece(m, cqt, chunk=64)
    np.testing.assert_allclose(full, chunked, atol=1e-5)


def test_norm_stats_and_buffers():
    pieces = [_synthetic_piece(seed=i) for i in range(2)]
    mean, std = compute_norm_stats(pieces)
    assert mean.shape == (88,) and std.shape == (88,) and np.all(std > 0)
    m = FrameCNN()
    m.set_norm(mean, std)
    assert torch.allclose(m.norm_mean[:, 0], torch.as_tensor(mean))


def test_sampler_shapes_and_alignment():
    p = _synthetic_piece(300)
    s = SegmentSampler([p], segment_frames=64, batch_size=4, seed=0)
    x, yf, yo = s.sample()
    assert x.shape == yf.shape == yo.shape == (4, 88, 64)
    assert set(torch.unique(yf).tolist()) <= {0.0, 1.0}


def test_training_learns_synthetic_task(tmp_path):
    cfg = Config()
    cfg.model.conv_channels = [8, 8, 8, 8]
    cfg.train.batch_size = 8
    cfg.train.segment_frames = 96
    cfg.train.learning_rate = 3e-3
    cfg.train.device = "cpu"
    # Enough material that every pitch is active somewhere in training: the
    # FC layer is pitch-specific, so a pitch never seen active scores
    # recall 0 (true of this architecture, irrelevant on real MAESTRO).
    train_p = [_synthetic_piece(n_frames=1500, seed=i, stem=f"t{i}") for i in range(10)]
    val_p = [_synthetic_piece(seed=100, stem="v")]
    out = train(cfg, train_p, val_p, tmp_path, epochs=4, steps_per_epoch=40, log=lambda *_: None)
    assert out["best_val_f1"] > 0.9
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()


def test_training_resumes_from_checkpoint(tmp_path):
    cfg = Config()
    cfg.model.conv_channels = [4, 4]
    cfg.train.batch_size = 4
    cfg.train.segment_frames = 64
    cfg.train.device = "cpu"
    train_p = [_synthetic_piece(seed=1)]
    val_p = [_synthetic_piece(seed=2)]
    train(cfg, train_p, val_p, tmp_path, epochs=1, steps_per_epoch=5, log=lambda *_: None)
    out = train(cfg, train_p, val_p, tmp_path, epochs=2, steps_per_epoch=5, log=lambda *_: None)
    assert [h["epoch"] for h in out["history"]] == [1, 2]
