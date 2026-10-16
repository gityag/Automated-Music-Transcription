"""
Make the README / paper figures from the JSON files in results/.

  fig1_ablation      note and note+offset F1 for run1-run4, MAESTRO test and SMD,
                     with 95% bootstrap intervals
  fig2_offsets       note+offset recall by note length, run2 vs run4, both test sets
  fig3_robustness    change in note F1 under each degradation, ours vs Kong et al.
  fig4_pianoroll     (optional, needs a checkpoint and the SMD cache) a 15 s excerpt:
                     reference notes vs the model's notes

Each figure is written as PNG (for the README) and PDF (for the paper).

Usage:
  pip install matplotlib
  python scripts/make_figures.py --results results --out figures \
      --ckpt models/run4_best.pt --smd-cache data/processed/smd
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette: categorical slots in fixed order, plus text / grid inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
REF_FILL = "#d9d8d2"

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def load(path: Path):
    return json.loads(path.read_text())


def save(fig, out: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"wrote {out / name}.png/.pdf")


def fig_ablation(res: Path, out: Path) -> None:
    runs = [("run1", "run1\nCNN"), ("run2", "run2\n+ onset head"), ("run3", "run3\n+ pitch GRU"),
            ("run4", "run4\n+ harm. conv")]
    metrics = [("note_onset_f1", "Note F1", BLUE), ("note_onset_offset_f1", "Note+offset F1", ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    x = np.arange(len(runs))
    w = 0.36
    for ax, (ds, title) in zip(axes, [("maestro_test", "MAESTRO v3 test (177 pieces)"),
                                       ("smd", "SMD (50 pieces, unseen recordings)")]):
        for k, (key, label, color) in enumerate(metrics):
            ci = [load(res / f"model_{ds}_{r}_ci.json")["ci"][key] for r, _ in runs]
            mean = np.array([c["mean"] for c in ci]) * 100
            err = np.array([[c["mean"] - c["lo"], c["hi"] - c["mean"]] for c in ci]).T * 100
            pos = x + (k - 0.5) * w
            ax.bar(pos, mean, width=w - 0.04, color=color, label=label, zorder=2)
            ax.errorbar(pos, mean, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
            for p, m, e in zip(pos, mean, err[1]):
                ax.text(p, m + e + 1.0, f"{m:.1f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x, [lbl for _, lbl in runs], fontsize=8.5)
        ax.set_title(title, color=INK, loc="left")
        ax.set_ylim(0, 105)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("F1 (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.02))
    fig.suptitle("The onset head fixes note starts; the per-pitch GRU fixes note endings",
                 x=0.06, y=1.0, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, "fig1_ablation")


def fig_offsets(res: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, (fname, title) in zip(axes, [("analysis_maestro_test.json", "MAESTRO v3 test"),
                                          ("analysis_smd.json", "SMD")]):
        d = load(res / fname)
        for run, label, color in [("run2", "run2: CNN + onset head", BLUE),
                                  ("run4", "run4: + per-pitch GRU, harmonic conv", ORANGE)]:
            rows = d["models"][run]["by_duration"]
            y = [r["onset_offset_recall"] * 100 for r in rows]
            ax.plot(range(len(rows)), y, color=color, linewidth=2, marker="o", markersize=5, label=label)
            ax.annotate(f"{y[-1]:.0f}", (len(rows) - 1, y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        labels = [r["bin"].replace(" s", "").replace(">=", "≥") for r in rows]
        ax.set_xticks(range(len(rows)), labels)
        ax.set_xlabel("Note length (s)")
        ax.set_title(f"{title} ({d['n_notes']:,} notes)", color=INK, loc="left")
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.3, len(rows) - 0.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Note+offset recall (%)")
    axes[0].legend(loc="lower left")
    fig.suptitle("Without memory over time, long notes lose their endings; the GRU recovers them",
                 x=0.06, ha="left", fontsize=12, color=INK)
    save(fig, out, "fig2_offsets")


def fig_robustness(res: Path, out: Path) -> None:
    t = load(res / "robustness_smd.json")
    conds = [("noise_20db", "Noise, 20 dB SNR"), ("noise_10db", "Noise, 10 dB SNR"),
             ("reverb_0.5s", "Reverb, RT60 0.5 s"), ("reverb_1.5s", "Reverb, RT60 1.5 s"),
             ("lowpass_4khz", "Low-pass 4 kHz"), ("detune_-25c", "Detune −25 cents"),
             ("detune_+25c", "Detune +25 cents")]
    systems = [("ours", "Ours (3.8M params)", BLUE), ("kong", "Kong et al. (42.9M)", ORANGE)]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = np.arange(len(conds))[::-1]
    h = 0.36
    for k, (s, label, color) in enumerate(systems):
        clean = t["clean"][s]["note_f1"]
        delta = np.array([(t[c][s]["note_f1"] - clean) * 100 for c, _ in conds])
        pos = y + (0.5 - k) * h
        ax.barh(pos, delta, height=h - 0.04, color=color, label=f"{label}, clean {clean * 100:.1f}", zorder=2)
        for p, v in zip(pos, delta):
            ax.text(v - 0.4 if v < 0 else v + 0.4, p, f"{v:+.1f}", va="center",
                    ha="right" if v < 0 else "left", fontsize=8, color=INK2)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y, [lbl for _, lbl in conds])
    ax.set_xlabel("Change in note F1 from clean audio (points)")
    ax.set_xlim(-28, 4)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2)
    ax.set_title("SMD, 50 pieces. Detune +50 cents (not shown): ours 46.0, Kong et al. 21.7 note F1",
                 loc="left", fontsize=9, color=INK2)
    fig.suptitle("The compact model tolerates detuning and a dull low-pass; reverb hurts it more",
                 x=0.02, ha="left", fontsize=12, color=INK)
    save(fig, out, "fig3_robustness")


def fig_pianoroll(ckpt: str, cache: Path, out: Path, stem: str, start: float, dur: float) -> None:
    import mir_eval

    from amt.data.cache import load_piece
    from amt.data.midi import Note
    from amt.evaluation.decode import decode_notes
    from amt.inference import default_thresholds, load_checkpoint
    from amt.models.cnn import predict_piece

    model, cfg, device = load_checkpoint(ckpt)
    fs = cfg.audio.sample_rate / cfg.cqt.hop_length
    piece = load_piece(cache / f"{stem}.npz", with_notes=True)
    fprob, oprob = predict_piece(model, piece.cqt, device=device)
    est = decode_notes(fprob, oprob, fs=fs, pitch_low=cfg.labels.pitch_low, **default_thresholds(model))
    ref = [Note(int(p), float(s), float(e), 100)
           for p, s, e in zip(piece.ref_pitches, piece.ref_onsets, piece.ref_offsets)]

    def iv(notes):
        return (np.array([[n.start, max(n.end, n.start + 1e-3)] for n in notes]),
                mir_eval.util.midi_to_hz(np.array([n.pitch for n in notes], dtype=float)))
    hit = np.zeros(len(est), dtype=bool)
    if est and ref:
        pairs = mir_eval.transcription.match_notes(*iv(ref), *iv(est), onset_tolerance=0.05, offset_ratio=None)
        for _, j in pairs:
            hit[j] = True

    t0, t1 = start, start + dur
    inwin = lambda n: n.end > t0 and n.start < t1  # noqa: E731
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for n in filter(inwin, ref):
        ax.add_patch(plt.Rectangle((n.start, n.pitch - 0.45), n.end - n.start, 0.9,
                                   facecolor=REF_FILL, edgecolor="none", zorder=1))
    for n, h in zip(est, hit):
        if inwin(n):
            ax.add_patch(plt.Rectangle((n.start, n.pitch - 0.3), n.end - n.start, 0.6, facecolor="none",
                                       edgecolor=BLUE if h else ORANGE, linewidth=1.2, zorder=2))
    pitches = [n.pitch for n in ref + est if inwin(n)] or [60]
    ax.set_xlim(t0, t1)
    ax.set_ylim(min(pitches) - 2, max(pitches) + 2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI pitch")
    ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=REF_FILL, edgecolor="none"),
               plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=BLUE, linewidth=1.2),
               plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=ORANGE, linewidth=1.2)]
    ax.legend(handles, ["Reference (pedal-extended)", "Predicted, onset matched", "Predicted, no match"],
              loc="upper left", bbox_to_anchor=(0, 1.13), ncol=3, fontsize=8)
    ax.set_title(f"{stem.replace('-SMD', '')}, {t0:.0f}-{t1:.0f} s", loc="right", fontsize=9, color=INK2)
    save(fig, out, "fig4_pianoroll")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--ckpt", default=None, help="checkpoint for the piano-roll figure")
    ap.add_argument("--smd-cache", default="data/processed/smd")
    ap.add_argument("--piece", default="Chopin_Op028-15_006_20100611-SMD")
    ap.add_argument("--start", type=float, default=20.0)
    ap.add_argument("--duration", type=float, default=15.0)
    args = ap.parse_args()

    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_ablation(res, out)
    fig_offsets(res, out)
    fig_robustness(res, out)
    if args.ckpt:
        fig_pianoroll(args.ckpt, Path(args.smd_cache), out, args.piece, args.start, args.duration)


if __name__ == "__main__":
    main()
