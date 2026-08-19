"""Figure 2 (framework / measurement principle) for the SpecEdge paper.

Archetype: schematic-led composite (nature-figure skill, Archetype 1).
  Top band (~50% height): pipeline schematic with real thumbnails --
    SEM -> frontend box -> predicted mask -> extractor box -> record card,
    plus the reference-free guard branch underneath.
  Bottom band: four measurement-principle panels computed from the real
    reference mask of one in-distribution sample --
    (b) cross-sections on one line component,
    (c) the two edge profiles and the width,
    (d) detrended width residual with the +/-3 sigma band (LWR),
    (e) one-sided edge PSD with the high-frequency half shaded.

All quantitative panels are computed live from the sample's reference mask;
nothing is sketched by hand.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / "paper" / "figures" / "fig2_framework.pdf"
OUT_PNG = ROOT / "paper" / "figures" / "fig2_framework.png"

SAMPLE = "00000007"

TEAL = "#0f766e"
VIOLET = "#7c5fc9"
AQUA_FILL = "#bfe3df"
BOX_FACE = "#f5f5f5"
BOX_EDGE = "0.35"
RED = "#c1272d"
GRAY = "0.45"

FIG_W, FIG_H = 7.16, 1.95


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (np.argmax(sizes) + 1)


def edge_profiles(comp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column top/bottom edge rows of a horizontal line component."""
    cols = np.where(comp.any(axis=0))[0]
    top = np.array([np.where(comp[:, c])[0].min() for c in cols], dtype=float)
    bot = np.array([np.where(comp[:, c])[0].max() for c in cols], dtype=float)
    return cols.astype(float), top, bot


def detrend(v: np.ndarray) -> np.ndarray:
    x = np.arange(v.size, dtype=float)
    coef = np.polyfit(x, v, 1)
    return v - np.polyval(coef, x)


def psd_one_sided(res: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    res = res - res.mean()
    win = np.hanning(res.size)
    spec = np.fft.rfft(res * win)
    power = np.abs(spec) ** 2
    freq = np.fft.rfftfreq(res.size, d=1.0)
    k = freq / freq.max()  # normalize to Nyquist
    return k[1:], power[1:]


def box(ax, x, y, w, h, title, lines, title_color="black") -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008",
        fc=BOX_FACE, ec=BOX_EDGE, lw=0.8, transform=ax.transAxes))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top",
            fontsize=6.8, fontweight="bold", color=title_color,
            transform=ax.transAxes)
    ax.text(x + w / 2, y + h / 2 - 0.028, lines, ha="center", va="center",
            fontsize=5.8, color="0.2", transform=ax.transAxes,
            linespacing=1.5)


def arrow(ax, x0, y0, x1, y1, color=GRAY) -> None:
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), transform=ax.transAxes, arrowstyle="-|>",
        mutation_scale=8, color=color, lw=1.0))


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                       "DejaVu Serif"],
        "axes.linewidth": 0.6,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
    })

    sem = load_gray(ROOT / "dataset/litho/images/test" / f"{SAMPLE}.png")
    gt = load_gray(ROOT / "dataset/litho/masks/test" / f"{SAMPLE}.png") > 127
    pred = load_gray(ROOT / "output/baselines/segformer/preds/masks" / f"{SAMPLE}.png") > 127

    comp = largest_component(gt)
    cols, top, bot = edge_profiles(comp)
    width = bot - top
    cd_mean = width.mean()
    w_res = detrend(width)
    lwr3 = 3.0 * w_res.std()
    t_res = detrend(top)
    k, power = psd_one_sided(t_res)
    hf_ratio = power[k >= 0.5].sum() / power.sum()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")

    # Zoom window into the line component, in SEM/GT 1024 coordinates.
    zx0, zx1 = 300, 620
    zy0, zy1 = 360, 500

    # ---------- measurement principle ----------
    bw, bh, by = 0.185, 0.62, 0.20
    bx = [0.055, 0.305, 0.555, 0.805]

    # (b) cross-sections on the component (SEM backdrop + mask fill + edges)
    ax_b = fig.add_axes([bx[0], by, bw, bh])
    ax_b.imshow(sem[zy0:zy1, zx0:zx1], cmap="gray", vmin=0, vmax=255,
                aspect="auto", extent=(zx0, zx1, zy1, zy0))
    mask_rgba = np.zeros((zy1 - zy0, zx1 - zx0, 4))
    mask_rgba[comp[zy0:zy1, zx0:zx1]] = matplotlib.colors.to_rgba(
        AQUA_FILL, alpha=0.35)
    ax_b.imshow(mask_rgba, aspect="auto", extent=(zx0, zx1, zy1, zy0),
                interpolation="nearest")
    sel = (cols >= zx0) & (cols < zx1)
    step = 40
    for c in cols[sel][::step]:
        ax_b.plot([c, c], [zy0 + 8, zy1 - 8], color="white", lw=0.5, ls=":")
    ax_b.plot(cols[sel], top[sel], color=TEAL, lw=1.1)
    ax_b.plot(cols[sel], bot[sel], color=VIOLET, lw=1.1)
    ax_b.set_xlabel("position along line $j$ (px)", fontsize=6)
    ax_b.set_ylabel("row (px)", fontsize=6)
    ax_b.set_title("(a) cross-sections $\\perp$ line", fontsize=6.8,
                   fontweight="bold")
    ax_b.tick_params(labelsize=5.5)
    for s in ax_b.spines.values():
        s.set_edgecolor(RED); s.set_linestyle("--"); s.set_linewidth(0.9)

    # (c) edge profiles and width
    ax_c = fig.add_axes([bx[1], by, bw, bh])
    ax_c.fill_between(cols, top, bot, color=AQUA_FILL, alpha=0.6, lw=0)
    ax_c.plot(cols, top, color=TEAL, lw=0.9)
    ax_c.plot(cols, bot, color=VIOLET, lw=0.9)
    ax_c.invert_yaxis()
    ax_c.text(cols[30], top[30] - 6, "edge $\\ell_j$", fontsize=6, color=TEAL,
              va="bottom")
    ax_c.text(cols[30], bot[30] + 6, "edge $r_j$", fontsize=6, color=VIOLET,
              va="top")
    ax_c.set_xlabel("position along line $j$ (px)", fontsize=6)
    ax_c.set_ylabel("edge row (px)", fontsize=6)
    ax_c.set_title("(b) edge profiles, width $w_j$", fontsize=6.8,
                   fontweight="bold")
    ax_c.text(0.5, 0.5, f"CD $= \\mathrm{{mean}}_j(w_j)$\n$= {cd_mean:.1f}$ px",
              transform=ax_c.transAxes, fontsize=6.2, ha="center", va="center")
    ax_c.tick_params(labelsize=5.5)

    # (d) detrended width residual -> LWR
    ax_d = fig.add_axes([bx[2], by, bw, bh])
    ax_d.plot(cols, w_res, color="0.25", lw=0.7)
    ax_d.axhline(3 * w_res.std(), color=TEAL, lw=0.9, ls="--")
    ax_d.axhline(-3 * w_res.std(), color=TEAL, lw=0.9, ls="--")
    ax_d.set_ylim(-3 * w_res.std() * 1.6, 3 * w_res.std() * 1.6)
    ax_d.set_xlabel("position along line $j$ (px)", fontsize=6)
    ax_d.set_ylabel("$\\widetilde{w}_j$ (px)", fontsize=6)
    ax_d.set_title("(c) detrended residual", fontsize=6.8, fontweight="bold")
    ax_d.text(0.97, 0.965, f"LWR $3\\sigma = {lwr3:.2f}$ px",
              transform=ax_d.transAxes, fontsize=6, va="top", ha="right")
    ax_d.text(0.97, 0.05, "LER: same statistic on one edge",
              transform=ax_d.transAxes, fontsize=5.5, color="0.4",
              va="bottom", ha="right")
    ax_d.tick_params(labelsize=5.5)

    # (e) one-sided edge PSD
    ax_e = fig.add_axes([bx[3], by, bw, bh])
    ax_e.loglog(k, power, color=TEAL, lw=0.8)
    ax_e.axvspan(0.5, 1.0, color="0.88", zorder=0)
    ax_e.set_xlabel("$k$ / Nyquist", fontsize=6)
    ax_e.set_ylabel("edge PSD $P(k)$", fontsize=6)
    ax_e.set_title("(d) edge PSD, HF half", fontsize=6.8, fontweight="bold")
    ax_e.text(0.05, 0.05, f"HF ratio $= {hf_ratio:.3f}$",
              transform=ax_e.transAxes, fontsize=6, ha="left", va="bottom")
    ax_e.tick_params(labelsize=5.5)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=300)
    print(f"CD={cd_mean:.2f}px LWR3s={lwr3:.2f}px HF={hf_ratio:.4f}")
    print(f"wrote {OUT_PDF}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
