"""Page-1 teaser: a directly predicted mask is not measurement-grade.

Three tiles rendered from one SEM field of the study dataset:

  left    -- the direct segmentation mask (blue), produced by
             scripts/prepare_teaser_sam_mask.py (box-prompted SAM ViT-H);
             the red box marks the zoom window
  middle  -- the zoom window: the direct mask's boundary sits pixels off
             the bright resist edge, and the linewidth arrow spans the
             correspondingly short mask extent
  right   -- the same window with the dataset's reference mask (teal),
             the output of the reference protocol's 1-D profile
             regression (arXiv:2511.12005); the arrow spans edge to edge

Inputs: dataset/SEM/ADI_train/<sample>.bmp (SEM field),
dataset/SEM/train_new_gt/<sample>.png (reference mask),
output/teaser/sam_direct_<sample>.png (direct mask).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from PIL import Image

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "fig_teaser"

SAMPLE = "00000040"
# zoom window in image coordinates: one line plus its end cap
WIN_Y0, WIN_Y1, WIN_X0, WIN_X1 = 330, 530, 680, 880
ARROW_COL = 30  # window column at which the linewidth arrow is read

BLUE = np.array([70, 125, 255])   # direct mask
TEAL = np.array([0, 205, 185])    # refined reference mask
MASK_ALPHA = 0.42


def tint(gray: np.ndarray, mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray] * 3, -1).astype(float)
    rgb[mask] = rgb[mask] * (1 - MASK_ALPHA) + color * MASK_ALPHA
    return rgb.astype(np.uint8)


def linewidth_arrow(ax, mask_window: np.ndarray) -> None:
    ys = np.where(mask_window[:, ARROW_COL])[0]
    y_top, y_bot = ys.min(), ys.max()
    ax.annotate("", xy=(ARROW_COL, y_top), xytext=(ARROW_COL, y_bot),
                arrowprops=dict(arrowstyle="<->", color="white", lw=0.9,
                                shrinkA=0, shrinkB=0))
    label = ax.text(ARROW_COL + 9, (y_top + y_bot) / 2, "linewidth",
                    fontsize=5.2, color="white", ha="left", va="center")
    label.set_path_effects([pe.withStroke(linewidth=1.2, foreground="black")])


def strip_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    image = np.asarray(Image.open(
        ROOT / "dataset" / "SEM" / "ADI_train" / f"{SAMPLE}.bmp").convert("L"))
    ref = np.asarray(Image.open(
        ROOT / "dataset" / "SEM" / "train_new_gt" / f"{SAMPLE}.png").convert("L")) > 127
    direct = np.asarray(Image.open(
        ROOT / "output" / "teaser" / f"sam_direct_{SAMPLE}.png").convert("L")) > 127

    fig = plt.figure(figsize=(3.5, 1.42))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.045,
                           left=0.005, right=0.995, top=0.845, bottom=0.01)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(tint(image, direct, BLUE))
    ax.add_patch(plt.Rectangle((WIN_X0, WIN_Y0), WIN_X1 - WIN_X0,
                               WIN_Y1 - WIN_Y0, fill=False, ec="red", lw=0.8))
    ax.set_title("segmentation mask, direct", fontsize=6.5,
                 fontweight="bold", pad=3)
    strip_axis(ax)

    window = image[WIN_Y0:WIN_Y1, WIN_X0:WIN_X1]
    tiles = [
        (direct, BLUE, "boxed window, zoomed", "boundary off the edge"),
        (ref, TEAL, "after 1-D refinement", "boundary on the edge"),
    ]
    for ci, (mask, color, title, note) in enumerate(tiles, start=1):
        ax = fig.add_subplot(gs[0, ci])
        mask_window = mask[WIN_Y0:WIN_Y1, WIN_X0:WIN_X1]
        ax.imshow(tint(window, mask_window, color))
        ax.contour(mask_window.astype(float), levels=[0.5],
                   colors=[color / 255], linewidths=0.9)
        linewidth_arrow(ax, mask_window)
        ax.set_title(title, fontsize=6.5, fontweight="bold", pad=3)
        text = ax.text(0.5, 0.03, note, transform=ax.transAxes, fontsize=5.4,
                       color="white", ha="center", va="bottom")
        text.set_path_effects([pe.withStroke(linewidth=1.0, foreground="black")])
        strip_axis(ax)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {"dpi": 600}), ("png", {"dpi": 300})):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", **kw)
        print(f"wrote {OUT}.{ext}")


if __name__ == "__main__":
    main()
