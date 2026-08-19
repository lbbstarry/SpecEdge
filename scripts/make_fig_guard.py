"""Figure: the reference-free guard, as a signal and as a tail bound.

Left  -- cross-frontend disagreement d against SegFormer CD MAE on both
         splits, with the threshold d* (the in-distribution 95th percentile,
         so no Extreme label enters it) and the failure level tau_sigma.
Right -- risk-coverage: accept the lowest-d fraction of Extreme samples and
         read the accepted set's worst and mean CD MAE. This is the paper's
         claim about the guard -- it bounds the tail, not the average.

Nothing is recomputed here: the curve comes from the CSV that
e4d_routing.py::risk_coverage already writes. The larger two-column versions
of these panels in e4d_routing.py and replot_paper_figures.py::fig_guard are
left untouched, since paper/supplementary/MOVED.md records them as the
rollback path for the two figures dropped for the page limit.

Run manually:
    python scripts/make_fig_guard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RV = ROOT / "output/revision_v4"
OUT = ROOT / "paper/figures/fig_guard"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman",
                              "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["legend.frameon"] = False

IN_DIST = "#5B7FBF"
EXTREME = "#C0392B"
NEUTRAL = "#555555"


def main() -> None:
    dis = pd.read_csv(RV / "e4_disagreement.csv")
    rc = pd.read_csv(RV / "e4d_routing_risk_coverage.csv")
    tau = json.load(open(RV / "e3_onset.json"))["tau_px"]
    dstar = json.load(open(RV / "e4d_routing.json"))["calibration"]["p95"]

    # One panel only: the scatter that used to sit beside it costs a third of
    # a column and says what the text already states about d*.
    fig, axr = plt.subplots(
        1, 1, figsize=(3.45, 1.18),
        gridspec_kw=dict(left=0.115, right=0.995, top=0.96, bottom=0.28))

    sf = dis[dis.model == "segformer"].dropna(
        subset=["disagreement", "abs_err_cd_mean"])
    n_flag = int((sf[sf.split == "extreme"]["disagreement"] > dstar).sum())

    ext = rc[rc.split == "extreme"].sort_values("coverage")
    axr.plot(ext["coverage"], ext["max_err_accepted"], lw=1.0, color=EXTREME,
             label="worst accepted")
    axr.plot(ext["coverage"], ext["mean_err_accepted"], lw=1.0, color=IN_DIST,
             ls="--", label="mean accepted")
    axr.axhline(tau, color=NEUTRAL, lw=0.8, ls=":")

    at85 = ext.iloc[(ext["coverage"] - 0.85).abs().to_numpy().argmin()]
    full = ext.iloc[-1]
    axr.plot([at85["coverage"]], [at85["max_err_accepted"]], marker="o",
             ms=3.0, color=EXTREME, zorder=4)
    axr.annotate(f"{at85['max_err_accepted']:.2f}", fontsize=5.6,
                 color=EXTREME, ha="right", va="top",
                 xy=(at85["coverage"], at85["max_err_accepted"]),
                 xytext=(-2, -3), textcoords="offset points")
    axr.annotate(f"{full['max_err_accepted']:.1f}", fontsize=5.6,
                 color=EXTREME, ha="right", va="bottom",
                 xy=(full["coverage"], full["max_err_accepted"]),
                 xytext=(-3, 3), textcoords="offset points")
    axr.text(0.502, tau * 1.25, "$\\tau_\\sigma$", fontsize=5.8, color=NEUTRAL,
             ha="left", va="bottom")
    axr.set_yscale("log")
    axr.set_xlim(0.48, 1.03)
    axr.set_xticks([0.5, 0.75, 0.85, 1.0])
    axr.set_xlabel("coverage (accept the lowest-$d$ fraction of Extreme)",
                   fontsize=6.5, labelpad=1.5)
    axr.set_ylabel("CD MAE (px)", fontsize=6.5, labelpad=1.5)
    axr.legend(fontsize=5.8, loc="upper left", handletextpad=0.4,
               borderaxespad=0.2, labelspacing=0.2)
    axr.tick_params(labelsize=5.8, length=2, pad=1.5)
    axr.spines[["top", "right"]].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(f"{OUT}.{suffix}", bbox_inches="tight", **kw)
        print(f"wrote {OUT}.{suffix}")


if __name__ == "__main__":
    main()
