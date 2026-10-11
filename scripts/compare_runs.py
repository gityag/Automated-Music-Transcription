"""
Compare two evaluation JSONs (from scripts/evaluate_model.py) scored on
the same folder, piece by piece, with a paired bootstrap.

Prints, per metric: mean of B minus A, its 95% interval, and p_le_zero
(the share of resamples where B is not better). An interval that excludes
0 means the difference is unlikely to come from which pieces happen to be
in the test set. It does NOT cover training noise: for that, compare a
seed repeat (e.g. run2 vs run2b) the same way.

Usage:
  python scripts/compare_runs.py results/model_maestro_test_run1.json results/model_maestro_test_run2.json
"""
from __future__ import annotations

import argparse
import json

from amt.evaluation.stats import paired_bootstrap

METRICS = {"frame F1": ("frame", "f1"),
           "note onset F1": ("note", "onset_f1"),
           "note onset+offset F1": ("note", "onset_offset_f1")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="baseline JSON")
    ap.add_argument("b", help="new JSON")
    args = ap.parse_args()

    a, b = (json.load(open(p)) for p in (args.a, args.b))
    for name, d in (("A", a), ("B", b)):
        if "per_piece" not in d:
            raise SystemExit(f"{name} has no per_piece scores: re-run evaluate_model.py on it")
    pa = {p["stem"]: p for p in a["per_piece"]}
    pb = {p["stem"]: p for p in b["per_piece"]}
    stems = sorted(pa.keys() & pb.keys())
    if len(stems) != len(pa) or len(stems) != len(pb):
        raise SystemExit(f"piece sets differ ({len(pa)} vs {len(pb)}, {len(stems)} shared)")

    print(f"B - A over {len(stems)} shared pieces\n  A = {args.a}\n  B = {args.b}")
    for label, (g, k) in METRICS.items():
        r = paired_bootstrap([pa[s][g][k] for s in stems], [pb[s][g][k] for s in stems])
        print(f"{label:22s} {100 * r['mean_diff']:+6.2f} pts  "
              f"95% CI [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}]  p_le_zero {r['p_le_zero']:.3f}")


if __name__ == "__main__":
    main()
