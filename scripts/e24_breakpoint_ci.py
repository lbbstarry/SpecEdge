"""E24: bootstrap intervals for the Extreme-split CD-MAE breakpoint.

The breakpoint estimator is of jump-changepoint type, so the n-out-of-n
bootstrap is not consistent for it: resampling all n points reproduces the
same grid value in almost every replicate and the percentile interval is
correspondingly narrow. This script quantifies that (the fraction of
replicates returning the estimate unchanged) and reports the m-out-of-n
interval at m = ceil(n**0.75), which is the one the paper reads as a region.

The two-piece fit is the same grid search revision_v4_analysis.py uses,
duplicated rather than imported because that module runs the whole revision
analysis at import time.

Run manually:
    python scripts/e24_breakpoint_ci.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output/hard_eval/segformer_metrology.csv"
OUT = ROOT / "output/revision_v4/e24_breakpoint_ci/summary.json"
SEED = 42
REPS = 2000


def two_piece_fit(x: np.ndarray, y: np.ndarray, min_side: int = 5):
    """Best split point of a broken-line fit, by grid search over midpoints."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    best = None
    ux = np.unique(xs)
    for i in range(len(ux) - 1):
        b = (ux[i] + ux[i + 1]) / 2
        left, right = xs <= b, xs > b
        if left.sum() < min_side or right.sum() < min_side:
            continue
        sse = 0.0
        for sel in (left, right):
            c = np.polyfit(xs[sel], ys[sel], 1)
            sse += float(((np.polyval(c, xs[sel]) - ys[sel]) ** 2).sum())
        if best is None or sse < best[1]:
            best = (b, sse)
    return best


def resample(x, y, size, reps, rng):
    """Breakpoints of `reps` bootstrap draws of `size` points with replacement."""
    out = []
    n = len(x)
    for _ in range(reps):
        idx = rng.integers(0, n, size)
        r = two_piece_fit(x[idx], y[idx])
        if r is not None:
            out.append(r[0])
    return np.asarray(out)


def main() -> None:
    df = pd.read_csv(SRC, dtype={"name": str}).dropna(
        subset=["abs_err_cd_mean", "gt_foreground_ratio"])
    x = df["gt_foreground_ratio"].to_numpy(float)
    y = np.log10(df["abs_err_cd_mean"].to_numpy(float) + 1e-3)
    n = len(x)
    bp, _ = two_piece_fit(x, y)

    rng = np.random.default_rng(SEED)
    b_n = resample(x, y, n, REPS, rng)
    m = int(round(n ** 0.75))
    b_m = resample(x, y, m, REPS, rng)

    def summarize(b, size):
        lo, hi = np.percentile(b, [2.5, 97.5])
        vals, counts = np.unique(np.round(b, 6), return_counts=True)
        return {
            "size": int(size), "reps": int(len(b)),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            # degeneracy of the resampling distribution: how concentrated the
            # replicate breakpoints are on the estimate and on few grid values
            "frac_at_estimate": round(float(np.mean(np.isclose(b, bp))), 4),
            "frac_modal": round(float(counts.max() / counts.sum()), 4),
            "frac_within_0.05": round(float(np.mean(np.abs(b - bp) <= 0.05)), 4),
            "n_distinct": int(len(vals)),
        }

    res = {
        "n": n, "seed": SEED, "breakpoint": round(float(bp), 6),
        "rule_for_m": "round(n**0.75)",
        "n_of_n": summarize(b_n, n),
        "m_of_n": summarize(b_m, m),
        "note": ("The estimator is of jump-changepoint type, so the n-out-of-n "
                 "bootstrap is inconsistent for it and its interval is "
                 "optimistic; the m-out-of-n interval is the reported one."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
