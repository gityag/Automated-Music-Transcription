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
from amt.training.loop import dilate_onsets, evaluate_frames, train


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


def test_dilate_onsets_widens_by_one_frame_each_side():
    y = torch.zeros(1, 1, 10)
    y[0, 0, 4] = 1
    assert dilate_onsets(y)[0, 0].tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert dilate_onsets(y, k=0).equal(y)


def test_onset_head_does_not_collapse_and_yields_notes(tmp_path):
    # Regression: with plain BCE the onset head collapsed to "never fires"
    # (max probability 0.006), giving note F1 of exactly 0 despite a good
    # frame head. pos_weight + target dilation must prevent that.
    from amt.data.midi import piano_roll_to_notes
    from amt.evaluation.decode import score_predictions

    cfg = Config()
    cfg.model.conv_channels = [8, 8, 8, 8]
    cfg.model.onset_head = True
    cfg.train.batch_size = 8
    cfg.train.segment_frames = 96
    cfg.train.learning_rate = 3e-3
    cfg.train.device = "cpu"
    train_p = [_synthetic_piece(n_frames=1500, seed=i, stem=f"t{i}") for i in range(10)]
    val = _synthetic_piece(n_frames=800, seed=99, stem="v")
    notes = piano_roll_to_notes(val.frame.astype(np.float32), val.onset.astype(np.float32), fs=62.5, pitch_low=21)
    val.ref_pitches = np.array([n.pitch for n in notes])
    val.ref_onsets = np.array([n.start for n in notes])
    val.ref_offsets = np.array([n.end for n in notes])

    train(cfg, train_p, [val], tmp_path, epochs=6, steps_per_epoch=40, log=lambda *_: None)
    model = build_model(cfg)
    model.load_state_dict(torch.load(tmp_path / "best.pt", weights_only=False)["model"])
    frame, onset = predict_piece(model, val.cqt, device="cpu")
    assert onset.max() > 0.5
    assert score_predictions([val], [(frame, onset)])["note"]["onset_f1"] > 0.6


def test_checkpoint_selected_by_note_f1_when_val_has_notes(tmp_path):
    from amt.data.midi import piano_roll_to_notes
    cfg = Config()
    cfg.model.conv_channels = [4, 4]
    cfg.model.onset_head = True
    cfg.train.batch_size = 4
    cfg.train.segment_frames = 64
    cfg.train.device = "cpu"
    val = _synthetic_piece(seed=3, stem="v")
    notes = piano_roll_to_notes(val.frame.astype(np.float32), val.onset.astype(np.float32), fs=62.5, pitch_low=21)
    val.ref_pitches = np.array([n.pitch for n in notes])
    val.ref_onsets = np.array([n.start for n in notes])
    val.ref_offsets = np.array([n.end for n in notes])
    out = train(cfg, [_synthetic_piece(seed=1)], [val], tmp_path, epochs=2, steps_per_epoch=5, log=lambda *_: None)
    h = out["history"][-1]
    assert h["select_metric"] == "note_onset_offset_f1"
    assert h["val_note_onset_offset_f1"] is not None
    ck = torch.load(tmp_path / "best.pt", weights_only=False)
    assert ck["select_metric"] == "note_onset_offset_f1"
    assert out["best_val_f1"] == pytest.approx(max(r["val_note_onset_offset_f1"] for r in out["history"]))


def test_checkpoint_falls_back_to_frame_f1_without_notes(tmp_path):
    cfg = Config()
    cfg.model.conv_channels = [4, 4]
    cfg.train.batch_size = 4
    cfg.train.segment_frames = 64
    cfg.train.device = "cpu"
    out = train(cfg, [_synthetic_piece(seed=1)], [_synthetic_piece(seed=2)], tmp_path,
                epochs=1, steps_per_epoch=5, log=lambda *_: None)
    assert out["history"][-1]["select_metric"] == "frame_f1"


def test_bigru_forward_shapes_and_param_count():
    x = torch.randn(2, 88, 64)
    m = FrameCNN(conv_channels=(4, 4), hidden=32, onset_head=True, temporal="bigru", pitch_feat=4, rnn_hidden=8)
    f, o = m(x)
    assert f.shape == (2, 88, 64) and o.shape == (2, 88, 64)
    # one GRU shared by all pitches: its size does not depend on n_bins
    assert sum(p.numel() for p in m.rnn.parameters()) == 2 * 3 * (8 * (4 + 2) + 8 * 8 + 2 * 8)
    with pytest.raises(ValueError):
        FrameCNN(temporal="lstm")


def test_bigru_starts_identical_to_cnn():
    # zero-initialised residual heads: before training, adding the GRU
    # changes nothing, so the GRU can only learn corrections
    torch.manual_seed(0)
    base = FrameCNN(conv_channels=(4, 4), hidden=16, onset_head=True).eval()
    torch.manual_seed(0)
    gru = FrameCNN(conv_channels=(4, 4), hidden=16, onset_head=True,
                   temporal="bigru", pitch_feat=4, rnn_hidden=8).eval()
    x = torch.randn(1, 88, 50)
    (bf, bo), (gf, go) = base(x), gru(x)
    assert torch.allclose(bf, gf) and torch.allclose(bo, go)


def test_bigru_chunked_prediction_matches_single_pass():
    torch.manual_seed(0)
    m = FrameCNN(conv_channels=(4, 4), hidden=16, onset_head=True,
                 temporal="bigru", pitch_feat=4, rnn_hidden=8).eval()
    cqt = np.random.default_rng(1).normal(-40, 10, size=(88, 3000)).astype(np.float32)
    m.set_norm(cqt.mean(axis=1), cqt.std(axis=1))  # as in training: unnormalized input saturates the GRU
    full_f, full_o = predict_piece(m, cqt, chunk=100_000)
    ch_f, ch_o = predict_piece(m, cqt, chunk=700)
    np.testing.assert_allclose(full_f, ch_f, atol=1e-4)
    np.testing.assert_allclose(full_o, ch_o, atol=1e-4)


def test_bigru_trains_on_synthetic_task(tmp_path):
    cfg = Config()
    cfg.model.conv_channels = [8, 8, 8, 8]
    cfg.model.onset_head = True
    cfg.model.temporal = "bigru"
    cfg.model.pitch_feat = 8
    cfg.model.rnn_hidden = 16
    cfg.train.batch_size = 8
    cfg.train.segment_frames = 96
    cfg.train.learning_rate = 3e-3
    cfg.train.device = "cpu"
    train_p = [_synthetic_piece(n_frames=1500, seed=i, stem=f"t{i}") for i in range(10)]
    out = train(cfg, train_p, [_synthetic_piece(seed=100, stem="v")], tmp_path,
                epochs=6, steps_per_epoch=40, log=lambda *_: None)
    assert out["best_val_f1"] > 0.85   # the CNN alone reaches ~0.93 here
    ck = torch.load(tmp_path / "best.pt", weights_only=False)
    assert ck["config"]["model"]["temporal"] == "bigru"
    m = build_model(cfg)
    m.load_state_dict(ck["model"])   # the checkpoint rebuilds from its own config


def test_harmonic_dilations_for_12_bins_per_octave():
    from amt.models.cnn import harmonic_dilations
    assert harmonic_dilations(12) == [12, 19, 24, 28, 31, 34, 36]


def test_harmonic_conv_connects_each_bin_to_its_harmonics():
    from amt.models.cnn import HarmonicDilatedConv
    conv = HarmonicDilatedConv(1, 1, [12, 19])
    with torch.no_grad():
        for c in [conv.base, *conv.harm]:
            c.weight.zero_()
            c.weight[0, 0, 0, 1] = 1.0      # centre time tap only
        conv.base.bias.zero_()
    x = torch.zeros(1, 1, 88, 5)
    x[0, 0, 60, 2] = 1.0                    # one active bin
    y = conv(x)[0, 0, :, 2]
    # bin 60 itself, and the bins whose 2nd / 3rd harmonic is bin 60
    assert torch.nonzero(y).flatten().tolist() == [60 - 19, 60 - 12, 60]


def test_harmonic_model_shapes_and_config():
    cfg = Config()
    cfg.model.conv_channels = [4, 4]
    cfg.model.onset_head = True
    cfg.model.harmonic = True
    m = build_model(cfg)
    assert m.harmonic is not None
    f, o = m(torch.randn(2, 88, 40))
    assert f.shape == (2, 88, 40) and o.shape == (2, 88, 40)
    assert build_model(Config()).harmonic is None     # off by default: old checkpoints load
