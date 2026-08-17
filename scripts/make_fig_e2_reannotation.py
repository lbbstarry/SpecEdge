"""E2 figure: blinded re-annotation of disputed regions (single-column).

Left column: two plate cells straight from the anonymized annotation kit --
one failure-sample card (large hallucinated blobs outlined in red) and one
control-sample crop (thin boundary slivers, the only disputed area healthy
samples produce). Right: stacked area-share bars per verdict for the two
groups; the control bar carries the 0/40 hallucination-verdict annotation.

Palette matches Figure 1: red_strong = hallucination, orange = missed by
reference, neutral = ambiguous boundary residue.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from PIL import Image

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["legend.frameon"] = False

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "output/revision_v4/reannotation_kit"
OUT = ROOT / "paper" / "figures" / "fig_e2_reannotation"

RED_STRONG = "#B64342"
ORANGE = "#E28E2C"
NEUTRAL = "#D8D8D8"
NEUTRAL_MID = "#767676"

VERDICT_COLORS = {
    "hallucination": RED_STRONG,
    "missed_by_reference": ORANGE,
    "ambiguous": NEUTRAL,
}
VERDICT_LABELS = {
    "hallucination": "hallucination",
    "missed_by_reference": "missed by reference",
    "ambiguous": "ambiguous (boundary)",
}


def densest_red_window(card_rgb: np.ndarray, win: int = 340) -> tuple[int, int]:
    """Top-left corner of the win x win crop with the most red contour pixels."""
    red = ((card_rgb[:, :, 0] > 180) & (card_rgb[:, :, 1] < 90)
           & (card_rgb[:, :, 2] < 90)).astype(float)
    best, best_rc = -1.0, (0, 0)
    step = 64
    for r in range(0, card_rgb.shape[0] - win, step):
        for c in range(0, card_rgb.shape[1] - win, step):
            s = red[r:r + win, c:c + win].sum()
            if s > best:
                best, best_rc = s, (r, c)
    return best_rc


def plate(ax, img: np.ndarray, label: str, sub: str, sub_color: str) -> None:
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.03, 0.955, label, transform=ax.transAxes, fontsize=6.4,
            fontweight="bold", color="white", va="top")
    ax.text(0.03, 0.06, sub, transform=ax.transAxes, fontsize=6,
            color=sub_color, va="bottom")


def main() -> None:
    summary = json.load(open(ROOT / "output/revision_v4/e2_reannotation_summary.json"))
    groups = summary["groups"]

    k01 = np.asarray(Image.open(KIT / "K01_B_components.png").convert("RGB"))
    k02 = np.asarray(Image.open(KIT / "K02_B_components.png").convert("RGB"))
    r0, c0 = densest_red_window(k02)
    k02_crop = k02[r0:r0 + 340, c0:c0 + 340]

    fig = plt.figure(figsize=(3.5, 3.1))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.15],
                           left=0.015, right=0.97, top=0.90, bottom=0.13,
                           hspace=0.14, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    plate(ax1, k01, "failure sample", "disputed regions", "#FFB4AE")
    ax2 = fig.add_subplot(gs[1, 0])
    plate(ax2, k02_crop, "control sample", "boundary slivers only", "#DDDDDD")

    # right: stacked area-share bars
    ax = fig.add_subplot(gs[:, 1])
    rows = [
        ("control_3", "control\n3 samples\n40 regions"),
        ("failure_5", "failure\n5 samples\n12 regions"),
    ]
    order = ["hallucination", "missed_by_reference", "ambiguous"]
    for yi, (gkey, ylabel) in enumerate(rows):
        verdicts = groups[gkey]["verdicts"]
        left = 0.0
        for v in order:
            frac = verdicts.get(v, {}).get("area_frac", 0.0)
            if frac <= 0:
                continue
            ax.barh(yi, frac, left=left, height=0.52,
                    color=VERDICT_COLORS[v], edgecolor="white", lw=0.5)
            left += frac
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[1] for r in rows], fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("share of disputed area", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fail = groups["failure_5"]["verdicts"]
    ax.text(fail["hallucination"]["area_frac"] / 2, 1.0,
            f"{fail['hallucination']['area_frac']*100:.1f}%\nhallucination",
            fontsize=6.2, color="white", ha="center", va="center",
            fontweight="bold")
    ax.text(0.5, 0.0, "100% ambiguous\n0/40 hallucination verdicts",
            fontsize=6.2, color="0.25", ha="center", va="center")

    handles = [plt.Line2D([], [], marker="s", ls="", mfc=VERDICT_COLORS[v],
                          mec="none", ms=5, label=VERDICT_LABELS[v])
               for v in order]
    ax.legend(handles=handles, fontsize=5.5, loc="upper center",
              bbox_to_anchor=(0.5, 1.18), ncol=1, handletextpad=0.4,
              labelspacing=0.25)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", **kw)
        print(f"wrote {OUT}.{ext}")


if __name__ == "__main__":
    main()
