"""
Training loop for FrameCNN.

An "epoch" here is a fixed number of random-window steps (by default one
expected pass over the training audio), not a sweep of a dataset object,
because training draws random windows from long pieces.

Checkpoints are written every epoch (last.pt, plus best.pt by validation
frame F1) and the loop resumes from last.pt, since Kaggle sessions can end
at any time.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from amt.data.cache import Piece, SegmentSampler, compute_norm_stats
from amt.models.cnn import build_model, predict_piece


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_frames(model, pieces: list[Piece], device, threshold: float = 0.5) -> dict:
    """Frame-level precision/recall/F1, with TP/FP/FN pooled over all pieces
    (so long pieces count in proportion to their length)."""
    tp = fp = fn = 0
    for p in pieces:
        probs, _ = predict_piece(model, p.cqt, device=device)
        pred = probs >= threshold
        ref = p.frame.astype(bool)
        tp += int(np.sum(pred & ref))
        fp += int(np.sum(pred & ~ref))
        fn += int(np.sum(~pred & ref))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def train(
    cfg,
    train_pieces: list[Piece],
    val_pieces: list[Piece],
    out_dir: str | Path,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    resume: bool = True,
    log=print,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tc = cfg.train
    epochs = epochs or tc.epochs
    device = pick_device(tc.device)
    torch.manual_seed(tc.seed)

    sampler = SegmentSampler(train_pieces, tc.segment_frames, tc.batch_size, seed=tc.seed)
    if steps_per_epoch is None:
        total_frames = sum(p.cqt.shape[1] for p in train_pieces)
        steps_per_epoch = max(1, total_frames // (tc.segment_frames * tc.batch_size))
    total_steps = epochs * steps_per_epoch

    model = build_model(cfg)
    mean, std = compute_norm_stats(train_pieces)
    model.set_norm(mean, std)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=tc.learning_rate)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=tc.learning_rate * 0.05)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict] = []
    best_f1, start_epoch = -1.0, 0
    last_path = out_dir / "last.pt"
    if resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        history, best_f1, start_epoch = ck["history"], ck["best_f1"], ck["epoch"]
        log(f"resumed from epoch {start_epoch} (best val F1 {best_f1:.4f})")

    log(f"device={device}  train={len(train_pieces)} pieces  val={len(val_pieces)} pieces  "
        f"epochs={epochs}  steps/epoch={steps_per_epoch}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    for epoch in range(start_epoch, epochs):
        sampler.reseed(tc.seed * 1000 + epoch)
        model.train()
        t0, running = time.time(), 0.0
        for step in range(steps_per_epoch):
            x, yf, yo = sampler.sample()
            x, yf, yo = x.to(device), yf.to(device), yo.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                frame_logits, onset_logits = model(x)
                loss = F.binary_cross_entropy_with_logits(frame_logits.float(), yf)
                if onset_logits is not None:
                    loss = loss + F.binary_cross_entropy_with_logits(onset_logits.float(), yo)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if (step + 1) % 100 == 0:
                log(f"  epoch {epoch+1} step {step+1}/{steps_per_epoch} loss {running/(step+1):.4f}")

        val = evaluate_frames(model, val_pieces, device)
        row = {"epoch": epoch + 1, "train_loss": running / steps_per_epoch,
               "val_precision": val["precision"], "val_recall": val["recall"], "val_f1": val["f1"],
               "seconds": time.time() - t0, "lr": sched.get_last_lr()[0]}
        history.append(row)
        log(f"epoch {epoch+1}/{epochs}  loss {row['train_loss']:.4f}  val P {val['precision']:.3f} "
            f"R {val['recall']:.3f} F1 {val['f1']:.3f}  ({row['seconds']:.0f}s)")

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch + 1,
                        "val_f1": best_f1}, out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "history": history, "best_f1": best_f1, "epoch": epoch + 1,
                    "config": asdict(cfg)}, last_path)
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))

    return {"best_val_f1": best_f1, "history": history}
