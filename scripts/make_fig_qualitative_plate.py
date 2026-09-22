"""Qualitative plate: two Extreme samples under the routed frontend pair.

Dark image plate, single-column: rows =
samples, columns = SEM+reference | DeepLabV3+ | SegFormer (the fallback and
the frontend the guard routes away from). Claim-only overlay grammar shared
with fig1: cyan = claimed and agreed, red = claimed and spurious; misses are
sub-percent boundary slivers on both samples and are not painted. Per-cell
CD error printed on the plate, red when above tau_sigma = 2.65 px.

Row story: #14 is an architecture-specific SegFormer collapse (CNNs at
<=0.35 px on the same image); #22 is read low by every frontend (~3.4 px
each) because all four claim a border-cut line the reference excludes --
model at fault above, reference at fault below.
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

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["legend.frameon"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "fig_qualitative_plate"

SAMPLES = ["14", "22"]
MODELS = ["deeplabv3plus", "segformer"]
COL_TITLES = ["SEM + reference", "DeepLabV3+", "SegFormer"]
VERDICTS = {"14": "model fails", "22": "reference misses"}
TAU = 2.65

RED_CALLOUT = "#E53935"
CYAN = "#22D7E6"


def norm_name(x: object) -> str:
    s = str(x)
    return s.lstrip("0") or "0"


def load_binary(path: Path, size: int = 1024) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.NEAREST)
    return np.asarray(img) > 127


def pick_window(gt: np.ndarray, preds: list[np.ndarray], size: int = 72,
                margin: int = 20, row_min: int = 0) -> tuple[int, int, int, int]:
    """One deterministic zoom window per sample: the size x size box near the
    reference boundary holding the most prediction-reference disagreement
    (union over frontends), so every column magnifies the same contested
    edge. row_min skips regions whose story the plate already tells
    elsewhere (the #22 border-cut strip)."""
    band = ndi.binary_dilation(gt ^ ndi.binary_erosion(gt), iterations=6)
    interest = np.zeros(gt.shape, dtype=float)
    for p in preds:
        interest += ((p ^ gt) & band)
    score = ndi.uniform_filter(interest, size=size, mode="constant")
    h = size // 2
    score[: max(margin, row_min) + h] = -1.0
    score[gt.shape[0] - margin - h:] = -1.0
    score[:, : margin + h] = -1.0
    score[:, gt.shape[1] - margin - h:] = -1.0
    r, c = np.unravel_index(int(np.argmax(score)), score.shape)
    return (r - h, r + h, c - h, c + h)


def edge_inset(ax, sem, pred, gt, win, label=False) -> None:
    """Corner magnifier over `win` in the tile's own overlay grammar, marked
    on the parent with a connector box (the Fig. 2 zoom idiom)."""
    r0, r1, c0, c1 = win
    axins = ax.inset_axes([0.55, 0.52, 0.44, 0.44])
    axins.imshow(sem, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
    if pred is not None:
        overlays = ((pred & gt, CYAN, 0.32), (pred & ~gt, RED_CALLOUT, 0.5))
    else:
        overlays = ((gt, CYAN, 0.32),)
    for mask, color, alpha in overlays:
        if mask.any():
            rgba = np.zeros((*mask.shape, 4))
            rgba[mask] = matplotlib.colors.to_rgba(color, alpha=alpha)
            axins.imshow(rgba, interpolation="bilinear")
    axins.set_xlim(c0, c1)
    axins.set_ylim(r1, r0)
    axins.set_xticks([])
    axins.set_yticks([])
    for s in axins.spines.values():
        s.set_visible(True)
        s.set_color("white")
        s.set_linewidth(0.7)
    ax.indicate_inset_zoom(axins, edgecolor="white", linewidth=0.6, alpha=0.9)
    if label:
        axins.text(0.95, 0.06, f"{0.44 * gt.shape[1] / (c1 - c0):.0f}$\\times$",
                   transform=axins.transAxes, fontsize=5, color="white",
                   ha="right", va="bottom")


def cell(ax, sem, pred, gt, note=None, note_red=False) -> None:
    ax.imshow(sem, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
    # translucent mask fills: cyan = agreed foreground (the reference mask
    # itself in the reference column), red / orange = the disagreement
    if pred is not None:
        overlays = ((pred & gt, CYAN, 0.32), (pred & ~gt, RED_CALLOUT, 0.5))
    else:
        overlays = ((gt, CYAN, 0.32),)
    for mask, color, alpha in overlays:
        if mask.any():
            rgba = np.zeros((*mask.shape, 4))
            rgba[mask] = matplotlib.colors.to_rgba(color, alpha=alpha)
            ax.imshow(rgba, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if note is not None:
        ax.text(0.04, 0.06, note, transform=ax.transAxes, fontsize=6,
                color="#FFB4AE" if note_red else "white", va="bottom")


def main() -> None:
    errs: dict[tuple[str, str], float] = {}
    for m in MODELS:
        df = pd.read_csv(ROOT / "output/hard_eval" / f"{m}_metrology.csv")
        df["key"] = df["name"].map(norm_name)
        for sid in SAMPLES:
            r = df[df["key"] == sid]
            errs[(m, sid)] = float(r["abs_err_cd_mean"].iloc[0])

    # One grid row per sample: the extra unused row left a blank band that
    # bbox_inches="tight" cannot trim, because the legend anchors below it.
    fig = plt.figure(figsize=(3.5, 2.62))
    gs = gridspec.GridSpec(len(SAMPLES), 3, figure=fig, hspace=0.05, wspace=0.04,
                           left=0.10, right=0.995, top=0.92, bottom=0.06)

    for ri, sid in enumerate(SAMPLES):
        # 512 metrology grid, matching the predictions' native resolution
        sem = np.asarray(Image.open(
            ROOT / "dataset/litho_hard/images/hard" / f"{sid}.png")
            .convert("L").resize((512, 512), Image.BILINEAR))
        gt = load_binary(ROOT / "dataset/litho_hard/masks/hard" / f"{sid}.png", 512)
        preds = {m: load_binary(
            ROOT / "output/hard_eval" / m / "preds/masks" / f"{sid}.png", 512)
            for m in MODELS}
        win = pick_window(gt, list(preds.values()),
                          row_min=48 if sid == "22" else 0)

        ax0 = fig.add_subplot(gs[ri, 0])
        cell(ax0, sem, None, gt)
        edge_inset(ax0, sem, None, gt, win, label=True)
        ax0.text(-0.13, 0.5, f"#{sid} · {VERDICTS[sid]}",
                 transform=ax0.transAxes, fontsize=6.0, fontweight="bold",
                 rotation=90, ha="center", va="center")
        if ri == 0:
            ax0.set_title(COL_TITLES[0], fontsize=7, fontweight="bold", pad=4)

        for ci, m in enumerate(MODELS, start=1):
            pred = preds[m]
            ax = fig.add_subplot(gs[ri, ci])
            e = errs[(m, sid)]
            cell(ax, sem, pred, gt,
                 note=f"CD err {e:.2f} px", note_red=(e > TAU))
            edge_inset(ax, sem, pred, gt, win)
            if ri == 0:
                ax.set_title(COL_TITLES[ci], fontsize=7, fontweight="bold",
                             pad=4)

    handles = [
        plt.Line2D([], [], marker="s", ls="", mfc=CYAN, mec="none",
                   ms=5, label=r"claimed foreground, agreed (pred $\cap$ ref)"),
        plt.Line2D([], [], marker="s", ls="", mfc=RED_CALLOUT, mec="none",
                   ms=5, label=r"claimed, spurious (pred $\setminus$ ref)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=5.4,
               bbox_to_anchor=(0.54, -0.03), handletextpad=0.4,
               columnspacing=0.9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {"dpi": 600}), ("png", {"dpi": 300})):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", **kw)
        print(f"wrote {OUT}.{ext}")


if __name__ == "__main__":
    main()
