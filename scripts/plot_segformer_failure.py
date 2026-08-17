"""Plot SegFormer failure profile: CD MAE vs GT foreground ratio on both splits."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/segformer_failure_analysis"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["unet", "deeplabv3plus", "hrnet", "segformer"]
COLORS = {"unet": "#1f77b4", "deeplabv3plus": "#2ca02c", "hrnet": "#ff7f0e", "segformer": "#d62728"}


def load(model: str, split: str) -> pd.DataFrame:
    if split == "standard":
        path = ROOT / f"output/metrology/{model}_test_metrics.csv"
    else:
        path = ROOT / f"output/hard_eval/{model}_metrology.csv"
    df = pd.read_csv(path, dtype={"name": str})
    return df.dropna(subset=["abs_err_cd_mean", "gt_foreground_ratio"])


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

for ax, split, title in zip(
    axes,
    ["standard", "hard"],
    ["Standard test (n=60)", "External process-critical set (n=65, complex removed → 62)"],
):
    for model in MODELS:
        df = load(model, split)
        ax.scatter(
            df["gt_foreground_ratio"], df["abs_err_cd_mean"],
            label=model, color=COLORS[model], alpha=0.7, s=40, edgecolors="white", linewidths=0.6,
        )
    ax.set_xlabel("GT foreground ratio")
    ax.set_title(title, fontsize=11)
    ax.set_yscale("symlog", linthresh=0.5)
    ax.axhline(1.0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.axhline(10.0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Per-sample |CD MAE| (px, symlog)")
axes[1].legend(loc="upper right", fontsize=9, framealpha=0.95)

# Highlight the SegFormer failure region
axes[1].axvspan(0.0, 0.25, alpha=0.08, color="red", label="_nolegend_")
axes[1].text(
    0.005, 30, "SegFormer\nfailure zone\n(fg < ~0.25)",
    fontsize=9, color="darkred", va="top",
)

plt.suptitle(
    "SegFormer fails on extreme-sparse-foreground samples that exist in the\n"
    "external set but not in the standard test split",
    fontsize=12, y=1.02,
)
plt.tight_layout()
plt.savefig(OUT / "segformer_failure_scatter.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}/segformer_failure_scatter.png")
