"""Page-1 teaser: a directly predicted mask is not measurement-grade.

Three tiles rendered from one SEM field of the study dataset:

  left    -- the direct segmentation mask (blue), produced by
             scripts/prepare_teaser_sam_mask.py (box-prompted SAM ViT-H);
             the red box marks the zoom window
  middle  -- the zoom window: the direct mask's boundary sits pixels
             inside the bright resist edge and is artificially smooth
             (edge magnifier inset); the linewidth arrow spans the
             correspondingly short mask extent
  right   -- the same window with the contour recovered by per-scanline
             one-dimensional profile regression (the reference-protocol
             recipe, arXiv:2511.12005): each scanline's boundary is the
             sub-pixel brightness peak near the reference boundary, so
             the contour (teal) is centered on the edge and carries the
             true edge roughness (inset); the arrow spans edge to edge

Inputs: dataset/litho/images/train/<sample>.png (SEM field),
dataset/litho/masks/train/<sample>.png (reference mask; supplies the
per-scanline search window for the profile extraction),
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
# zoom window in image coordinates: a straight line span, left of its end cap
# (the per-column extraction is undefined where the boundary turns vertical)
WIN_Y0, WIN_Y1, WIN_X0, WIN_X1 = 335, 520, 615, 800
ARROW_COL = 25   # window column at which the linewidth arrow is read
LINE_W = 0.4     # contour line width, pt

# edge magnifier: source strip (window coordinates) and inset placement
INSET_X0, INSET_X1, INSET_H = 92, 128, 16
INSET_RECT = (0.55, 0.045, 0.43, 0.22)

BLUE = np.array([70, 125, 255])   # direct mask
TEAL = np.array([0, 205, 185])    # 1-D profile contour
FILL_ALPHA_FIELD = 0.42
FILL_ALPHA_ZOOM = 0.22

MIN_RUN_PX = 25    # skip columns whose mask run is shorter (cap region)
PEAK_OUT_PX = 4    # search from this far outside the reference boundary...
PEAK_IN_PX = 10    # ...to this far inside it


def subpixel_peak(profile: np.ndarray, lo: int, hi: int) -> float:
    """Brightness peak in [lo, hi), refined by parabolic interpolation."""
    segment = profile[lo:hi]
    p = int(np.argmax(segment))
    y = float(lo + p)
    if 0 < p < len(segment) - 1:
        a, b, c = segment[p - 1], segment[p], segment[p + 1]
        denom = a - 2 * b + c
        if denom != 0:
            y += 0.5 * (a - c) / denom
    return y


def profile_chains(image: np.ndarray, ref: np.ndarray):
    """Per-column 1-D profile edges (top and bottom) inside the window."""
    top, bot = [], []
    for x in range(WIN_X0, WIN_X1):
        ys = np.where(ref[:, x])[0]
        ys = ys[(ys >= WIN_Y0 - 30) & (ys < WIN_Y1 + 30)]
        if len(ys) < MIN_RUN_PX:
            top.append(np.nan)
            bot.append(np.nan)
            continue
        t, b = ys.min(), ys.max()
        prof = image[:, x]
        top.append(subpixel_peak(prof, max(0, t - PEAK_OUT_PX), t + PEAK_IN_PX + 1))
        bot.append(subpixel_peak(prof, b - PEAK_IN_PX, min(image.shape[0], b + PEAK_OUT_PX + 1)))
    return np.array(top) - WIN_Y0, np.array(bot) - WIN_Y0


def tint(gray: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    rgb = np.stack([gray] * 3, -1).astype(float)
    rgb[mask] = rgb[mask] * (1 - alpha) + color * alpha
    return rgb.astype(np.uint8)


def add_magnifier(ax, window: np.ndarray, center_y: float, draw) -> None:
    y0 = int(round(center_y - INSET_H / 2))
    y1 = y0 + INSET_H
    ax.add_patch(plt.Rectangle((INSET_X0, y0), INSET_X1 - INSET_X0, INSET_H,
                               fill=False, ec="white", lw=0.5, ls=(0, (2, 1.2))))
    axi = ax.inset_axes(INSET_RECT)
    axi.imshow(window[y0:y1, INSET_X0:INSET_X1], cmap="gray", vmin=0, vmax=255,
               extent=(INSET_X0 - 0.5, INSET_X1 - 0.5, y1 - 0.5, y0 - 0.5),
               interpolation="nearest", aspect="auto")
    draw(axi)
    axi.set_xlim(INSET_X0 - 0.5, INSET_X1 - 0.5)
    axi.set_ylim(y1 - 0.5, y0 - 0.5)
    axi.set_xticks([])
    axi.set_yticks([])
    for spine in axi.spines.values():
        spine.set_visible(True)
        spine.set_color("white")
        spine.set_linewidth(0.6)
        spine.set_linestyle((0, (2, 1.2)))


def linewidth_arrow(ax, y_top: float, y_bot: float) -> None:
    ax.annotate("", xy=(ARROW_COL, y_top), xytext=(ARROW_COL, y_bot),
                arrowprops=dict(arrowstyle="<->", color="white", lw=0.8,
                                shrinkA=0, shrinkB=0))
    label = ax.text(ARROW_COL + 7, (y_top + y_bot) / 2, "linewidth",
                    fontsize=5.0, color="white", ha="left", va="center")
    label.set_path_effects([pe.withStroke(linewidth=1.1, foreground="black")])


def note(ax, text: str) -> None:
    t = ax.text(0.03, 0.03, text, transform=ax.transAxes, fontsize=5.2,
                color="white", ha="left", va="bottom", linespacing=1.15)
    t.set_path_effects([pe.withStroke(linewidth=1.0, foreground="black")])


def strip_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    image = np.asarray(Image.open(
        ROOT / "dataset" / "litho" / "images" / "train" / f"{SAMPLE}.png").convert("L")).astype(float)
    ref = np.asarray(Image.open(
        ROOT / "dataset" / "litho" / "masks" / "train" / f"{SAMPLE}.png").convert("L")) > 127
    direct = np.asarray(Image.open(
        ROOT / "output" / "teaser" / f"sam_direct_{SAMPLE}.png").convert("L")) > 127

    image8 = image.astype(np.uint8)
    window = image8[WIN_Y0:WIN_Y1, WIN_X0:WIN_X1]
    top_chain, bot_chain = profile_chains(image, ref)
    xs = np.arange(WIN_X1 - WIN_X0)

    fig = plt.figure(figsize=(3.5, 1.42))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.045,
                           left=0.005, right=0.995, top=0.845, bottom=0.01)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(tint(image8, direct, BLUE, FILL_ALPHA_FIELD))
    ax.add_patch(plt.Rectangle((WIN_X0, WIN_Y0), WIN_X1 - WIN_X0,
                               WIN_Y1 - WIN_Y0, fill=False, ec="red", lw=0.8))
    ax.set_title("segmentation mask, direct", fontsize=6.5,
                 fontweight="bold", pad=3)
    strip_axis(ax)

    # middle: the direct mask, smooth and biased inward
    ax = fig.add_subplot(gs[0, 1])
    mask_window = direct[WIN_Y0:WIN_Y1, WIN_X0:WIN_X1]
    ax.imshow(tint(window, mask_window, BLUE, FILL_ALPHA_ZOOM))
    ax.contour(mask_window.astype(float), levels=[0.5],
               colors=[BLUE / 255], linewidths=LINE_W)
    ys = np.where(mask_window[:, ARROW_COL])[0]
    linewidth_arrow(ax, float(ys.min()), float(ys.max()))
    edge_y = np.array([np.where(mask_window[:, x])[0].min()
                       for x in range(INSET_X0, INSET_X1) if mask_window[:, x].any()])
    add_magnifier(ax, window, edge_y.mean(),
                  lambda axi: axi.contour(mask_window.astype(float), levels=[0.5],
                                          colors=[BLUE / 255], linewidths=0.9))
    ax.set_title("boxed window, zoomed", fontsize=6.5, fontweight="bold", pad=3)
    note(ax, "smooth,\ninside the edge")
    ax.set_xlim(-0.5, WIN_X1 - WIN_X0 - 0.5)
    ax.set_ylim(WIN_Y1 - WIN_Y0 - 0.5, -0.5)
    strip_axis(ax)

    # right: the per-scanline 1-D profile contour
    ax = fig.add_subplot(gs[0, 2])
    fill = np.zeros_like(window, dtype=bool)
    for i in xs:
        if np.isnan(top_chain[i]) or np.isnan(bot_chain[i]):
            continue
        fill[int(round(top_chain[i])):int(round(bot_chain[i])) + 1, i] = True
    ax.imshow(tint(window, fill, TEAL, FILL_ALPHA_ZOOM))
    ax.plot(xs, top_chain, color=TEAL / 255, lw=LINE_W)
    ax.plot(xs, bot_chain, color=TEAL / 255, lw=LINE_W)
    linewidth_arrow(ax, top_chain[ARROW_COL], bot_chain[ARROW_COL])
    add_magnifier(ax, window, np.nanmean(top_chain[INSET_X0:INSET_X1]),
                  lambda axi: axi.plot(xs, top_chain, color=TEAL / 255, lw=0.9))
    ax.set_title("after 1-D refinement", fontsize=6.5, fontweight="bold", pad=3)
    note(ax, "rough, centered\non the edge")
    ax.set_xlim(-0.5, WIN_X1 - WIN_X0 - 0.5)
    ax.set_ylim(WIN_Y1 - WIN_Y0 - 0.5, -0.5)
    strip_axis(ax)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {"dpi": 600}), ("png", {"dpi": 300})):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", **kw)
        print(f"wrote {OUT}.{ext}")


if __name__ == "__main__":
    main()
