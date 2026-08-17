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

FIG_W, FIG_H = 7.16, 5.0


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
        "font.family": "DejaVu Sans",
        "axes.linewidth": 0.6,
        "mathtext.fontset": "dejavusans",
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

    # ---------- top band: pipeline schematic ----------
    thumb_h = 0.23
    thumb_w = thumb_h * FIG_H / FIG_W
    ty = 0.72

    ax_sem = fig.add_axes([0.015, ty, thumb_w, thumb_h])
    ax_sem.imshow(sem, cmap="gray", vmin=0, vmax=255)
    ax_sem.set_xticks([]); ax_sem.set_yticks([])
    ax_sem.set_title("SEM image $I$", fontsize=6.8, fontweight="bold", pad=2)

    box(canvas, 0.225, ty, 0.155, thumb_h,
        "Segmentation frontend $f_\\phi$",
        "any binary segmenter")

    ax_mask = fig.add_axes([0.43, ty, thumb_w, thumb_h])
    ax_mask.imshow(pred, cmap="gray", vmin=0, vmax=1)
    ax_mask.set_xticks([]); ax_mask.set_yticks([])
    ax_mask.set_title("mask $\\widehat{M}$", fontsize=6.8, fontweight="bold", pad=2)

    box(canvas, 0.635, ty, 0.155, thumb_h,
        "Mask-to-metrology extractor",
        "rule-based, deterministic\n(steps in b\u2013e)")

    box(canvas, 0.845, ty, 0.145, thumb_h,
        "Metrology record $\\mathbf{m}$",
        "CD mean / std\nLWR $3\\sigma$, LER $3\\sigma$\nPSD HF ratio\n$\\Delta N$, topology flags")

    ymid = ty + thumb_h / 2
    arrow(canvas, 0.015 + thumb_w + 0.005, ymid, 0.22, ymid)
    arrow(canvas, 0.385, ymid, 0.425, ymid)
    arrow(canvas, 0.43 + thumb_w + 0.005, ymid, 0.63, ymid)
    arrow(canvas, 0.795, ymid, 0.84, ymid)

    # ---------- offline qualification lane (Contribution 1) ----------
    qy = 0.565
    canvas.text(0.015, qy + 0.05, "offline\nqualification",
                fontsize=6, color="0.35", style="italic",
                ha="left", va="center", linespacing=1.4)
    box(canvas, 0.405, qy, 0.15, 0.10,
        "reference record $\\mathbf{m}_{\\mathrm{ref}}$",
        "qualification data only")
    box(canvas, 0.60, qy, 0.20, 0.10, "measurement-error scoring",
        "$|\\mathbf{m} - \\mathbf{m}_{\\mathrm{ref}}|$ vs. "
        "$\\sigma_{\\mathrm{ref}}$, across windows")
    box(canvas, 0.845, qy, 0.145, 0.10, "qualify frontend",
        "bounded operating region", title_color=TEAL)
    arrow(canvas, 0.56, qy + 0.05, 0.595, qy + 0.05)
    arrow(canvas, 0.805, qy + 0.05, 0.84, qy + 0.05)
    arrow(canvas, 0.9175, ty - 0.005, 0.70, qy + 0.105)

    # ---------- online monitoring lane (Contribution 2) ----------
    gy = 0.425
    canvas.text(0.015, gy + 0.05, "online\nmonitoring",
                fontsize=6, color="0.35", style="italic",
                ha="left", va="center", linespacing=1.4)
    box(canvas, 0.225, gy, 0.155, 0.10, "co-deployed frontends",
        "$\\widehat{M}_1 \\ldots \\widehat{M}_K$")
    box(canvas, 0.43, gy, 0.19, 0.10, "disagreement",
        "$d_m = 1 - \\mathrm{mean}_{o \\neq m}\\,\\mathrm{IoU}(\\widehat{M}_m, \\widehat{M}_o)$")
    box(canvas, 0.67, gy, 0.155, 0.10, "runtime guard",
        "$d_m > d^{\\star}$: flag / route", title_color=RED)
    arrow(canvas, 0.3025, ty - 0.005, 0.3025, gy + 0.105)
    arrow(canvas, 0.385, gy + 0.05, 0.425, gy + 0.05)
    arrow(canvas, 0.625, gy + 0.05, 0.665, gy + 0.05)

    canvas.text(0.015, 0.99, "(a)", fontsize=8, fontweight="bold", va="top")

    # ---------- red dashed zoom callout on the mask thumbnail ----------
    zx0, zx1 = 300, 620   # columns of the zoom window (SEM/GT 1024 coords)
    zy0, zy1 = 360, 500   # rows
    sx = pred.shape[1] / sem.shape[1]
    sy = pred.shape[0] / sem.shape[0]
    ax_mask.add_patch(Rectangle((zx0 * sx, zy0 * sy), (zx1 - zx0) * sx,
                                (zy1 - zy0) * sy,
                                fill=False, ec=RED, lw=0.9, ls="--"))

    # ---------- bottom band: measurement principle ----------
    bw, bh, by = 0.185, 0.26, 0.062
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
    ax_b.set_title("(b) cross-sections $\\perp$ line", fontsize=6.8,
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
    ax_c.set_title("(c) edge profiles, width $w_j$", fontsize=6.8,
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
    ax_d.set_title("(d) detrended residual", fontsize=6.8, fontweight="bold")
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
    ax_e.set_title("(e) edge PSD, HF half", fontsize=6.8, fontweight="bold")
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
