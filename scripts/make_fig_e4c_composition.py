"""E4c figure: guard AUROC across ensemble compositions (single-column).

Horizontal bars, sorted. Every ensemble containing the monitored SegFormer
stays within AUROC 0.900-0.928 on SegFormer-Extreme failure detection; the
CNN-only ensemble collapses to 0.525, near chance. One glance carries both
the robustness claim and the common-mode-failure caveat.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "paper" / "figures" / "fig_e4c_composition.pdf"
OUT_PNG = ROOT / "paper" / "figures" / "fig_e4c_composition.png"

TEAL = "#0f766e"
RED = "#c1272d"

NAMES = {
    "unet": "U-Net",
    "deeplabv3plus": "DLV3+",
    "hrnet": "HRNet",
    "segformer": "SegF.",
}


def label_of(ensemble: str) -> str:
    parts = [NAMES[p] for p in ensemble.split("+")]
    return " + ".join(parts)


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.6,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
    })

    df = pd.read_csv(ROOT / "output/revision_v4/e4c_loo_guard.csv")
    with_sf = df[(df["scope"] == "extreme_segformer")
                 & df["ensemble"].str.contains("segformer")].copy()
    cnn_only = df[(df["ensemble"] == "unet+deeplabv3plus+hrnet")
                  & (df["scope"] == "extreme_all_in_ensemble")].copy()

    rows = [(label_of(r["ensemble"]), r["auroc"], TEAL)
            for _, r in with_sf.iterrows()]
    rows += [(label_of(r["ensemble"]) + "\n(no SegFormer)", r["auroc"], RED)
             for _, r in cnn_only.iterrows()]
    rows.sort(key=lambda t: t[1])

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    y = range(len(rows))
    ax.barh(y, values, color=colors, height=0.62)
    for i, v in enumerate(values):
        ax.text(v + 0.008, i, f"{v:.3f}", fontsize=6, va="center")
    ax.axvline(0.5, color="0.4", lw=0.8, ls="--")
    ax.set_ylim(-0.55, len(rows) - 0.45 + 0.75)
    ax.text(0.5, len(rows) - 0.45 + 0.55, "chance", fontsize=6, color="0.4",
            ha="center", va="top")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("AUROC, SegFormer-on-Extreme failures", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT_PDF}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
