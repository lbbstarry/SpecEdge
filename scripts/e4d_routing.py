"""E4d: guard-triggered fallback routing and risk-coverage analysis.

Closes the loop on the disagreement guard: when the monitored frontend's
disagreement d exceeds a threshold d* (calibrated as a percentile of its
in-distribution disagreement, i.e. without touching Extreme labels), the
sample's metrology is routed to a fallback frontend. We report:

  1. Routed vs unrouted SegFormer CD MAE on Extreme across fallbacks.
  2. A risk-coverage curve: accept the lowest-d fraction c of samples,
     report max / mean CD MAE among the accepted set.
  3. The in-distribution cost of routing (flag rate, MAE change).

Pure pandas on cached per-sample CSVs; no GPU, no mask recomputation.

Outputs:
  output/revision_v4/e4d_routing.json
  output/revision_v4/e4d_routing_risk_coverage.csv
  output/revision_v4/fig_e4d_routing.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from revision_v4_analysis import OUT  # noqa: E402

PFIG = OUT.parent.parent / "paper" / "figures"

MONITORED = "segformer"
FALLBACKS = ["hrnet", "unet", "deeplabv3plus"]
CAL_PCTS = [90, 95]


def load_disagreement() -> pd.DataFrame:
    df = pd.read_csv(OUT / "e4_disagreement.csv", dtype={"name": str})
    return df.dropna(subset=["abs_err_cd_mean"])


def routed_errors(df: pd.DataFrame, split: str, fallback: str,
                  d_star: float) -> dict:
    mon = df[(df.split == split) & (df.model == MONITORED)].set_index("name")
    fb = df[(df.split == split) & (df.model == fallback)].set_index("name")
    common = mon.index.intersection(fb.index)
    mon, fb = mon.loc[common], fb.loc[common]
    flagged = mon["disagreement"] > d_star
    routed = np.where(flagged, fb["abs_err_cd_mean"], mon["abs_err_cd_mean"])
    base = mon["abs_err_cd_mean"].to_numpy(float)
    return {
        "split": split, "fallback": fallback, "n": int(len(common)),
        "flag_rate": round(float(flagged.mean()), 4),
        "cd_mae_before": round(float(base.mean()), 4),
        "cd_mae_after": round(float(routed.mean()), 4),
        "cd_mae_max_before": round(float(base.max()), 4),
        "cd_mae_max_after": round(float(routed.max()), 4),
        "cd_p90_before": round(float(np.percentile(base, 90)), 4),
        "cd_p90_after": round(float(np.percentile(routed, 90)), 4),
    }


def risk_coverage(df: pd.DataFrame, split: str) -> pd.DataFrame:
    mon = df[(df.split == split) & (df.model == MONITORED)].sort_values(
        "disagreement")
    err = mon["abs_err_cd_mean"].to_numpy(float)
    n = len(mon)
    rows = []
    for k in range(5, n + 1):
        acc = err[:k]
        rows.append({"split": split, "coverage": round(k / n, 4),
                     "n_accepted": k,
                     "max_err_accepted": round(float(acc.max()), 4),
                     "mean_err_accepted": round(float(acc.mean()), 4)})
    return pd.DataFrame(rows)


def main() -> None:
    df = load_disagreement()
    mon_std = df[(df.split == "standard") & (df.model == MONITORED)]

    results = {"monitored": MONITORED, "calibration": {}, "routing": []}
    for pct in CAL_PCTS:
        d_star = float(np.percentile(mon_std["disagreement"], pct))
        results["calibration"][f"p{pct}"] = round(d_star, 5)
        for split in ["extreme", "standard"]:
            for fallback in FALLBACKS:
                r = routed_errors(df, split, fallback, d_star)
                r["d_star_pct"] = pct
                results["routing"].append(r)

    rc = pd.concat([risk_coverage(df, s) for s in ["extreme", "standard"]])
    rc.to_csv(OUT / "e4d_routing_risk_coverage.csv", index=False)
    json.dump(results, open(OUT / "e4d_routing.json", "w"), indent=2)

    print(f"== E4d: guard-triggered routing (monitored={MONITORED}) ==")
    for pct in CAL_PCTS:
        print(f"\n  d* = in-dist P{pct} = {results['calibration'][f'p{pct}']}")
        for r in results["routing"]:
            if r["d_star_pct"] != pct:
                continue
            print(f"    {r['split']:<9s} -> {r['fallback']:<14s} "
                  f"flag={r['flag_rate']:.0%}  "
                  f"MAE {r['cd_mae_before']:.3f}->{r['cd_mae_after']:.3f}  "
                  f"max {r['cd_mae_max_before']:.1f}->{r['cd_mae_max_after']:.1f}  "
                  f"p90 {r['cd_p90_before']:.2f}->{r['cd_p90_after']:.2f}")

    # Sized for a two-column figure* at 7.2 in and styled to match
    # scripts/replot_paper_figures.py, so type renders at the same size as
    # the body text instead of being scaled down on inclusion.
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for split, c in [("extreme", "#d65f5f"), ("standard", "#4878d0")]:
        sub = rc[rc.split == split]
        axes[0].plot(sub["coverage"], sub["max_err_accepted"], "-", c=c,
                     label=f"{split}: max CD MAE among accepted")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("coverage (fraction of samples accepted by guard)")
    axes[0].set_ylabel("max CD MAE among accepted (px)")
    axes[0].legend(fontsize=8)

    sweep = []
    ext_mon = df[(df.split == "extreme") & (df.model == MONITORED)]
    for d_star in np.quantile(mon_std["disagreement"], np.linspace(0.5, 1.0, 26)):
        r = routed_errors(df, "extreme", "hrnet", float(d_star))
        sweep.append({"flag_rate": r["flag_rate"], "after": r["cd_mae_after"]})
    sw = pd.DataFrame(sweep)
    axes[1].plot(sw["flag_rate"], sw["after"], "-o", ms=3, c="#d65f5f",
                 label="routed to HRNet")
    axes[1].axhline(float(ext_mon["abs_err_cd_mean"].mean()), ls="--",
                    c="gray", lw=1, label="no routing")
    axes[1].set_xlabel("flag rate on Extreme")
    axes[1].set_ylabel("SegFormer Extreme CD MAE after routing (px)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PFIG / "fig_e4d_routing.pdf", bbox_inches="tight")  # vector: the paper includes this one
    fig.savefig(OUT / "fig_e4d_routing.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  outputs: {OUT / 'e4d_routing.json'}, fig_e4d_routing.png")


if __name__ == "__main__":
    main()
