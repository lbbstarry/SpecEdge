"""Figure: Extreme-split behavior of each frontend across 5-fold retraining.

One dot per (frontend, fold): HD95 on the left with the collapse level
HD95 = 15 drawn in, CD MAE on the right. The point the table version could
only imply with mean +/- sd is visible directly: SegFormer's five dots sit in
a tight cluster above the collapse line (it fails every time, CV 0.06), while
the CNNs scatter across it (they fail intermittently, on a schedule set by
the training draw). Exact values stay tabulated in the supplementary
material.

Reads the fold artifacts the CV run (e18) wrote; computes nothing new.

Run manually, like the other make_fig*.py scripts:
    python scripts/make_fig_cv.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "output/cv"
OUT = ROOT / "paper/figures/fig_cv"
N_FOLDS = 5
HD95_COLLAPSE = 15.0

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman",
                              "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["legend.frameon"] = False

# palette and order shared with make_fig1_overview.py
MODELS = [
    ("unet", "U-Net", "#484878"),
    ("deeplabv3plus", "DLv3+", "#7884B4"),
    ("hrnet", "HRNet", "#B4C0E4"),
    ("segformer", "SegF", "#B64342"),
]
JITTER = [-0.16, -0.08, 0.0, 0.08, 0.16]  # fold 0..4, same offset both panels


def load() -> pd.DataFrame:
    rows = []
    for k in range(N_FOLDS):
        for model, _, _ in MODELS:
            base = CV / f"fold{k}" / model
            hd = json.loads((base / "eval_hard.json").read_text())[
                "summary"]["hausdorff95"]
            cd = pd.read_csv(base / "metrology_hard.csv")["abs_err_cd_mean"] \
                .dropna().mean()
            rows.append({"model": model, "fold": k,
                         "hd95": float(hd), "cd": float(cd)})
    return pd.DataFrame(rows)


def main() -> None:
    df = load()

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(3.45, 1.30),
        gridspec_kw=dict(left=0.105, right=0.995, top=0.965, bottom=0.155,
                         wspace=0.42))

    for ax, key, ylabel in ((axl, "hd95", "Extreme HD95"),
                            (axr, "cd", "Extreme CD MAE (px)")):
        for xi, (model, _, colour) in enumerate(MODELS):
            sub = df[df.model == model].sort_values("fold")
            xs = [xi + JITTER[int(f)] for f in sub["fold"]]
            ax.scatter(xs, sub[key], s=9, c=colour, linewidths=0.4,
                       edgecolors="0.25", zorder=3)
            ax.plot([xi - 0.24, xi + 0.24], [sub[key].mean()] * 2,
                    color=colour, lw=1.1, zorder=2)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([short for _, short, _ in MODELS])
        ax.set_xlim(-0.55, len(MODELS) - 0.45)
        ax.set_ylabel(ylabel, fontsize=6.5, labelpad=1.5)
        ax.tick_params(labelsize=5.8, length=2, pad=1.5)
        ax.spines[["top", "right"]].set_visible(False)

    axl.axhline(HD95_COLLAPSE, color="0.35", lw=0.8, ls=(0, (3, 2)), zorder=1)
    axl.text(1.5, HD95_COLLAPSE - 0.9, "collapse", fontsize=5.4,
             color="0.35", ha="center", va="top")
    axl.set_ylim(0, 27)
    axr.set_ylim(0, 2.9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(f"{OUT}.{suffix}", bbox_inches="tight", **kw)
        print(f"wrote {OUT}.{suffix}")


if __name__ == "__main__":
    main()
