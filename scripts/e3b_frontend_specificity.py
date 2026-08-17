"""E3b: frontend specificity of the changepoint localization methodology.

Applies the same Extreme-only two-segment regression + bootstrap CI + F-test
used for SegFormer (E3) to all four frontends. The point is specificity: the
methodology should localize an operationally meaningful breakpoint only where
one exists, i.e. the below-breakpoint error regime should exceed the noise
floor tau only for the frontend that actually fails.

Outputs:
  output/revision_v4/e3b_frontend_specificity.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from revision_v4_analysis import (  # noqa: E402
    MODELS, OUT, RNG, metro_df, two_piece_fit,
)


def load_tau() -> float:
    with open(OUT / "e1b_noise_floor.json") as f:
        nf = json.load(f)
    return float(nf["extreme"]["systematic_1px"]["cd_mean"]["mean"])


def analyze(model: str, tau: float) -> dict:
    df = metro_df("extreme", model).dropna(
        subset=["abs_err_cd_mean", "gt_foreground_ratio"])
    x = df["gt_foreground_ratio"].to_numpy(float)
    err = df["abs_err_cd_mean"].to_numpy(float)
    y = np.log10(err + 1e-3)
    bp, sse2 = two_piece_fit(x, y)
    c1 = np.polyfit(x, y, 1)
    sse1 = float(((np.polyval(c1, x) - y) ** 2).sum())
    n = len(x)
    fstat = ((sse1 - sse2) / 3) / (sse2 / (n - 5))
    pval = float(stats.f.sf(fstat, 3, n - 5))
    boots = []
    for _ in range(1000):
        idx = RNG.integers(0, n, n)
        r = two_piece_fit(x[idx], y[idx])
        if r is not None:
            boots.append(r[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    below, above = x <= bp, x > bp
    return {
        "model": model, "n": n,
        "breakpoint": round(float(bp), 4),
        "breakpoint_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "f_test_p": pval,
        "n_below": int(below.sum()), "n_above": int(above.sum()),
        "cd_mae_below": round(float(err[below].mean()), 4),
        "cd_mae_above": round(float(err[above].mean()), 4),
        "below_exceeds_tau": bool(err[below].mean() > tau),
        "n_fail": int((err > tau).sum()),
    }


def main() -> None:
    tau = load_tau()
    print(f"== E3b: changepoint specificity across frontends (tau={tau:.2f} px) ==")
    results = [analyze(m, tau) for m in MODELS]
    json.dump({"tau_px": tau, "results": results},
              open(OUT / "e3b_frontend_specificity.json", "w"), indent=2)
    for r in results:
        print(f"  {r['model']:<14s} bp={r['breakpoint']:.3f} "
              f"CI[{r['breakpoint_ci95'][0]:.2f},{r['breakpoint_ci95'][1]:.2f}] "
              f"F-p={r['f_test_p']:.2e}  MAE below/above="
              f"{r['cd_mae_below']:.3f}/{r['cd_mae_above']:.3f} px  "
              f"below>tau={r['below_exceeds_tau']}  n_fail={r['n_fail']}")


if __name__ == "__main__":
    main()
