"""Qualitative plate: three Extreme samples under all four frontends.

Pattern-13 dark image plate (nature-figure skill): rows = samples, columns =
SEM+reference | U-Net | DeepLabV3+ | HRNet | SegFormer. Consistent crop
geometry and overlay colors across the grid; per-cell CD error printed on
the plate, red when above the noise floor tau_sigma = 2.65 px.

Row story: #14 and #10 are architecture-specific SegFormer collapses
(CNNs at <=0.35 px on the same image); #22 is hard for every frontend
(~3.4 px each) -- the plate shows both failure modes side by side.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from PIL import Image
from scipy import ndimage as ndi

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["legend.frameon"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "fig_qualitative_plate"

SAMPLES = ["14", "10", "22"]
MODELS = ["unet", "deeplabv3plus", "hrnet", "segformer"]
COL_TITLES = ["SEM + reference", "U-Net", "DeepLabV3+", "HRNet", "SegFormer"]
TAU = 2.65

RED_CALLOUT = "#E53935"
ORANGE = "#E28E2C"
CYAN = "#22D7E6"


def norm_name(x: object) -> str:
    s = str(x)
    return s.lstrip("0") or "0"


def load_binary(path: Path, size: int = 1024) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.NEAREST)
    return np.asarray(img) > 127


def cell(ax, sem, pred, gt, note=None, note_red=False) -> None:
    ax.imshow(sem, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
    if pred is not None:
        for mask, color in ((pred & ~gt, RED_CALLOUT), (gt & ~pred, ORANGE)):
            if mask.any():
                rgba = np.zeros((*mask.shape, 4))
                rgba[mask] = matplotlib.colors.to_rgba(color, alpha=0.5)
                ax.imshow(rgba, interpolation="nearest")
    boundary = gt ^ ndi.binary_erosion(gt, iterations=2)
    rgba = np.zeros((*boundary.shape, 4))
    rgba[boundary] = matplotlib.colors.to_rgba(CYAN, alpha=0.9)
    ax.imshow(rgba, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if note is not None:
        ax.text(0.04, 0.06, note, transform=ax.transAxes, fontsize=6,
                color="#FFB4AE" if note_red else "white", va="bottom")


def main() -> None:
    errs: dict[tuple[str, str], float] = {}
    fgs: dict[str, float] = {}
    for m in MODELS:
        df = pd.read_csv(ROOT / "output/hard_eval" / f"{m}_metrology.csv")
        df["key"] = df["name"].map(norm_name)
        for sid in SAMPLES:
            r = df[df["key"] == sid]
            errs[(m, sid)] = float(r["abs_err_cd_mean"].iloc[0])
            fgs[sid] = float(r["gt_foreground_ratio"].iloc[0])

    fig = plt.figure(figsize=(7.16, 4.45))
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.05, wspace=0.04,
                           left=0.055, right=0.995, top=0.93, bottom=0.015)

    for ri, sid in enumerate(SAMPLES):
        sem = np.asarray(Image.open(
            ROOT / "dataset/litho_hard/images/hard" / f"{sid}.png").convert("L"))
        gt = load_binary(ROOT / "dataset/litho_hard/masks/hard" / f"{sid}.png")

        ax0 = fig.add_subplot(gs[ri, 0])
        cell(ax0, sem, None, gt)
        ax0.text(-0.06, 0.5, f"#{sid} · fg {fgs[sid]:.2f}",
                 transform=ax0.transAxes, fontsize=6.5, fontweight="bold",
                 rotation=90, ha="center", va="center")
        if ri == 0:
            ax0.set_title(COL_TITLES[0], fontsize=7, fontweight="bold", pad=4)

        for ci, m in enumerate(MODELS, start=1):
            pred = load_binary(
                ROOT / "output/hard_eval" / m / "preds/masks" / f"{sid}.png")
            ax = fig.add_subplot(gs[ri, ci])
            e = errs[(m, sid)]
            cell(ax, sem, pred, gt,
                 note=f"CD err {e:.2f} px", note_red=(e > TAU))
            if ri == 0:
                ax.set_title(COL_TITLES[ci], fontsize=7, fontweight="bold",
                             pad=4)

    handles = [
        plt.Line2D([], [], marker="s", ls="", mfc=RED_CALLOUT, mec="none",
                   ms=5, label="spurious foreground (pred ∖ ref)"),
        plt.Line2D([], [], marker="s", ls="", mfc=ORANGE, mec="none",
                   ms=5, label="missed foreground (ref ∖ pred)"),
        plt.Line2D([], [], color=CYAN, lw=1.4, label="reference boundary"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6,
               bbox_to_anchor=(0.53, -0.035), handletextpad=0.4,
               columnspacing=1.2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", **kw)
        print(f"wrote {OUT}.{ext}")


if __name__ == "__main__":
    main()
