"""Figure 1 (teaser/overview) for the SpecEdge paper -- v2, nature-figure skill.

Archetype: image plate (hero) + supporting quant row.
  Hero strip (a), ~55% height, Pattern-13 plate styling: one Extreme sample
  (#9, the 48.2 px worst case) under three frontends, labels on the plate,
  no fake flow arrows -- the three outcome cells are alternatives on the
  same input, not a sequence.
  Support row: (b) IoU-metrology decoupling, (c) OOD x low-fg collapse,
  (d) reference-free guard + routing. Quieter than the hero per Pattern 12.

Palette: one baseline family (#484878/#7884B4/#B4C0E4) for CNN frontends and
the in-distribution split; red (#B64342/#E53935) reserved for failure
semantics (SegFormer identity, Extreme split, hallucination); cyan #22D7E6
for the reference boundary (imaging-plate accent); orange #E28E2C for
missed area and the breakpoint family.

All numbers come from per-sample artifacts in output/; nothing re-estimated.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from scipy import ndimage as ndi

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["legend.frameon"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "fig1_overview"

SAMPLE = "9"  # Extreme worst case: SegFormer CD err 48.2 px

# --- palette (nature-figure skill) ---
BASE_DARK = "#484878"   # U-Net
BASE_MID = "#7884B4"    # DeepLabV3+ / in-dist
BASE_SOFT = "#B4C0E4"   # HRNet
RED_STRONG = "#B64342"  # SegFormer / Extreme
RED_CALLOUT = "#E53935" # hallucination overlay
ORANGE = "#E28E2C"      # missed overlay / breakpoint family
CYAN = "#22D7E6"        # reference boundary
BG_AQUA = "#E0F0F0"     # IoU-bin shading
BG_PEACH = "#F0E0D0"    # breakpoint CI band
NEUTRAL_MID = "#767676"

MODEL_COLORS = {
    "unet": BASE_DARK,
    "deeplabv3plus": BASE_MID,
    "hrnet": BASE_SOFT,
    "segformer": RED_STRONG,
}
MODEL_LABELS = {
    "unet": "U-Net",
    "deeplabv3plus": "DeepLabV3+",
    "hrnet": "HRNet",
    "segformer": "SegFormer",
}


def norm_name(x: object) -> str:
    s = str(x)
    return s.lstrip("0") or "0"


def load_binary(path: Path, size: int = 1024) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.NEAREST)
    return np.asarray(img) > 127


def plate_cell(ax, sem, pred, gt, label, verdict, verdict_color) -> None:
    """One hero cell: SEM + overlays + on-plate labels (Pattern 13)."""
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
    ax.text(0.03, 0.965, label, transform=ax.transAxes, fontsize=6.6,
            fontweight="bold", color="white", va="top")
    ax.text(0.03, 0.055, verdict, transform=ax.transAxes, fontsize=6.2,
            color=verdict_color, va="bottom")


def panel_a(fig, gs_hero) -> None:
    sem = np.asarray(
        Image.open(ROOT / "dataset/litho_hard/images/hard" / f"{SAMPLE}.png").convert("L"))
    gt = load_binary(ROOT / "dataset/litho_hard/masks/hard" / f"{SAMPLE}.png")
    otsu = load_binary(
        ROOT / "output/revision_v4/e11_classical/otsu/preds/masks/extreme_hard" / f"{SAMPLE}.png")
    segf = load_binary(ROOT / "output/hard_eval/segformer/preds/masks" / f"{SAMPLE}.png")
    hrnet = load_binary(ROOT / "output/hard_eval/hrnet/preds/masks" / f"{SAMPLE}.png")

    # the in-distribution IoUs are qualification scores, so say so on the tile
    cells = [
        (otsu, "Threshold (industry)", "misses the line · IoU 0.14", "#FFB4AE"),
        (segf, "SegFormer · in-dist IoU 0.986", "hallucination · CD err 48.2 px",
         "#FFB4AE"),
        (hrnet, "HRNet · in-dist IoU 0.984", "accurate · CD err 0.07 px",
         "#A8E6B8"),
    ]
    axes = []
    inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_hero,
                                             wspace=0.045)
    for i, (pred, label, verdict, vc) in enumerate(cells):
        ax = fig.add_subplot(inner[i])
        plate_cell(ax, sem, pred, gt, label, verdict, vc)
        axes.append(ax)
    zoom_cell(fig.add_subplot(inner[3]), sem, gt)

    # scale bar on the first cell (pixel units; nm calibration pending)
    axes[0].plot([1024 - 60 - 200, 1024 - 60], [1024 - 55, 1024 - 55],
                 color="white", lw=1.6)
    axes[0].text(1024 - 60 - 100, 1024 - 85, "200 px", color="white",
                 fontsize=5.5, ha="center", va="bottom")

    # shared overlay legend under the strip
    handles = [
        plt.Line2D([], [], marker="s", ls="", mfc=RED_CALLOUT, mec="none",
                   ms=5, label=r"spurious foreground (pred $\setminus$ ref)"),
        plt.Line2D([], [], marker="s", ls="", mfc=ORANGE, mec="none",
                   ms=5, label=r"missed foreground (ref $\setminus$ pred)"),
        plt.Line2D([], [], color=CYAN, lw=1.4, label="reference boundary"),
    ]
    axes[0].legend(handles=handles, loc="upper left",
                   bbox_to_anchor=(0.0, -0.015), fontsize=6, ncol=3,
                   handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0)


def zoom_cell(ax, sem: np.ndarray, gt: np.ndarray) -> None:
    """Edge-level view of the paper's thesis: two displacement fields with the
    same L1 norm, drawn on this sample's upper edge. Overlap cannot separate
    them; CD and LER read them oppositely. The printed numbers are the c = 3
    population values of the controlled test (E23), not per-strip estimates."""
    e23 = json.load(open(ROOT / "output/revision_v4/e23_mechanism/summary.json"))
    arms = e23["by_c"]["3.0"]["arms"]
    d_iou = e23["by_c"]["3.0"]["iou_spread_across_arms"]
    cd_const = arms["constant"]["cd_err_mean"]
    cd_rough = arms["gaussian"]["cd_err_mean"]
    ler_up = arms["gaussian"]["ler_pert_mean"] - arms["gaussian"]["ler_ref_mean"]

    # metrology grid is 512; work there so 1 drawn px = 1 reported px
    g = np.asarray(Image.fromarray(gt).resize((512, 512), Image.NEAREST)) > 0
    s = np.asarray(Image.fromarray(sem).convert("L").resize((512, 512)))
    x0, x1 = 150, 362
    band = np.where(g[:, x0:x1].mean(1) > 0.5)[0]
    upper = np.array([np.where(g[:, x])[0].min() for x in range(x0, x1)], float)
    r0, r1 = int(band.min()) - 16, int(band.min()) + 13  # strip around the edge
    c = 3.0
    rng = np.random.default_rng(0)
    rough = upper - rng.normal(0.0, c * np.sqrt(np.pi / 2.0), upper.size)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # strip occupies the middle of the tile; map (x, row) into axes coords
    SY0, SY1 = 0.24, 0.80
    ax.imshow(s[r0:r1 + 1, x0:x1], cmap="gray", vmin=0, vmax=255,
              extent=(0.0, 1.0, SY0, SY1), aspect="auto",
              interpolation="nearest", zorder=1)
    def ty(rows):  # image row -> axes y
        return SY1 - (np.asarray(rows) - r0) / (r1 - r0) * (SY1 - SY0)
    xs = np.linspace(0.0, 1.0, upper.size)
    ax.plot(xs, ty(upper), color=CYAN, lw=1.1, zorder=3)
    ax.plot(xs, ty(upper - c), color=RED_CALLOUT, lw=1.0, zorder=3)
    ax.plot(xs, ty(rough), color="#5B7FBF", lw=0.7, alpha=0.95, zorder=2)

    ax.text(0.03, 0.965, "equal-$|\\delta|$ edge fields (controlled)",
            fontsize=6.6, fontweight="bold", va="top")
    ax.text(0.97, 0.845, f"one edge, zoomed · same IoU "
            f"($\\Delta \\leq {d_iou:.3f}$)", fontsize=5.6,
            color=NEUTRAL_MID, ha="right", va="bottom")
    ax.text(0.03, 0.175,
            f"offset $+c$: CD err {cd_const:.1f} px · LER unchanged",
            fontsize=5.9, color=RED_CALLOUT, va="top")
    ax.text(0.03, 0.075,
            f"rough, same $|\\delta|$: CD err {cd_rough:.1f} px · "
            f"LER $+{ler_up:.1f}$ px",
            fontsize=5.9, color="#5B7FBF", va="top")
    ax.text(0.03, ty(upper[0]) + 0.035, "reference", fontsize=5.4, color=CYAN,
            va="bottom")


def panel_b(ax) -> None:
    rows = []
    for model in MODEL_COLORS:
        per = json.load(open(ROOT / "output/baselines" / model / "eval_test.json"))["per_sample"]
        iou = {norm_name(r["name"]): r["iou"] for r in per}
        met = pd.read_csv(ROOT / "output/metrology" / f"{model}_test_metrics.csv")
        met["key"] = met["name"].map(norm_name)
        for _, r in met.iterrows():
            v = r.get("abs_err_cd_mean")
            if r["key"] in iou and pd.notna(v):
                rows.append((model, iou[r["key"]], float(v)))
    df = pd.DataFrame(rows, columns=["model", "iou", "cd"])
    df["cd"] = df["cd"].clip(lower=1e-3)

    # 232 of the 240 records sit in [0.980, 0.995]; starting the axis at 0.965
    # spent half its width on 8 points and hid the spread the panel is about.
    XLO, XHI = 0.9795, 0.9955
    BINS = [(0.980, 0.985), (0.985, 0.990), (0.990, 0.995)]

    n_clip = int((df["iou"] < XLO).sum())
    iou_min = float(df["iou"].min())
    for i, (lo, hi) in enumerate(BINS):  # alternate shading marks the bins
        if i % 2 == 1:
            ax.axvspan(lo, hi, color=BG_AQUA, zorder=0)
    # distinct markers as well as colours: the three CNN blues are close in
    # value, and at print size hue alone does not separate them
    markers = {"unet": "o", "deeplabv3plus": "s", "hrnet": "^",
               "segformer": "D"}
    for model in MODEL_COLORS:  # family order; SegFormer drawn last, on top
        sub = df[df["model"] == model]
        ax.scatter(sub["iou"], sub["cd"], s=7, c=MODEL_COLORS[model],
                   marker=markers[model], label=MODEL_LABELS[model],
                   alpha=0.8, linewidths=0)

    # one p5-p95 bar per bin, with the ratio the text quotes printed above it
    for lo, hi in BINS:
        b = df[(df["iou"] >= lo) & (df["iou"] < hi)]["cd"]
        p5, p95 = np.percentile(b, [5, 95])
        x = (lo + hi) / 2
        ax.plot([x, x], [p5, p95], color="black", lw=1.0, zorder=4)
        for y in (p5, p95):
            ax.plot([x - 5e-4, x + 5e-4], [y, y], color="black", lw=1.0,
                    zorder=4)
        ax.text(x, p95 * 2.0, f"{p95 / p5:.0f}$\\times$", fontsize=6.5,
                fontweight="bold", ha="center", va="bottom", zorder=4)

    ax.set_xlim(XLO, XHI)
    ax.set_ylim(6e-4, 4e3)  # headroom for the legend and the ratio labels
    ax.set_xticks([0.980, 0.985, 0.990, 0.995])
    ax.text(0.015, 0.02,
            f"$\\leftarrow$ {n_clip} of {len(df)} records off-axis, "
            f"down to IoU {iou_min:.2f}",
            transform=ax.transAxes, fontsize=5.5, color=NEUTRAL_MID,
            va="bottom")
    ax.text(0.985, 0.055, "bars: p5--p95 within bin", transform=ax.transAxes,
            fontsize=5.5, color="black", ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("per-sample IoU", fontsize=7)
    ax.set_ylabel("CD MAE (px)", fontsize=7)
    ax.set_title("(b) IoU does not rank metrology", fontsize=7.5,
                 fontweight="bold")
    ax.legend(fontsize=5.5, loc="upper left", handletextpad=0.1,
              borderaxespad=0.2, labelspacing=0.2, ncol=2, columnspacing=0.8)
    ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_c(ax) -> None:
    onset = json.load(open(ROOT / "output/revision_v4/e3_onset.json"))
    tau, bp = onset["tau_px"], onset["breakpoint"]
    # band = the m-out-of-n interval the paper adopts, not the n-out-of-n one
    # in e3_onset.json, which the text reports only to reject as optimistic
    ci = json.load(open(
        ROOT / "output/revision_v4/e24_breakpoint_ci/summary.json"))["m_of_n"]["ci95"]

    ext = pd.read_csv(ROOT / "output/hard_eval/segformer_metrology.csv").dropna(
        subset=["abs_err_cd_mean", "gt_foreground_ratio"])
    ind = pd.read_csv(ROOT / "output/metrology/segformer_test_metrics.csv").dropna(
        subset=["abs_err_cd_mean", "gt_foreground_ratio"])

    ax.axvspan(ci[0], ci[1], color=BG_PEACH, zorder=0)
    ax.axvline(bp, color=ORANGE, lw=1.0, ls="--")
    ax.axhline(tau, color=NEUTRAL_MID, lw=0.9, ls=":")
    ax.scatter(ind["gt_foreground_ratio"], ind["abs_err_cd_mean"].clip(lower=1e-3),
               s=7, c=BASE_MID, alpha=0.8, linewidths=0, label="in-dist")
    ax.scatter(ext["gt_foreground_ratio"], ext["abs_err_cd_mean"].clip(lower=1e-3),
               s=7, c=RED_STRONG, alpha=0.8, linewidths=0, label="Extreme")

    ext["key"] = ext["name"].map(norm_name)
    s9 = ext[ext["key"] == SAMPLE]
    ax.scatter(s9["gt_foreground_ratio"], s9["abs_err_cd_mean"], marker="*",
               s=85, c=RED_STRONG, edgecolors="black", linewidths=0.5, zorder=5)
    ax.annotate("sample (a)", (float(s9["gt_foreground_ratio"].iloc[0]),
                               float(s9["abs_err_cd_mean"].iloc[0])),
                textcoords="offset points", xytext=(6, -2), fontsize=6)
    # horizontal, above the band: rotated inside the shading it was unreadable
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 12)
    ax.text(bp, 0.975, f"breakpoint {bp:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]",
            transform=ax.get_xaxis_transform(), fontsize=6, color=ORANGE,
            ha="center", va="top")
    ax.text(0.745, tau * 1.5, "$\\tau_\\sigma$ = 2.65 px", fontsize=6,
            color=NEUTRAL_MID, ha="right")
    ax.set_xlabel("foreground ratio", fontsize=7)
    ax.set_ylabel("SegFormer CD MAE (px)", fontsize=7)
    ax.set_title("(c) Collapse: OOD × low fg", fontsize=7.5, fontweight="bold")
    ax.legend(fontsize=5.5, loc="upper right", handletextpad=0.1,
              labelspacing=0.2)
    ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_d(ax) -> None:
    dis = pd.read_csv(ROOT / "output/revision_v4/e4_disagreement.csv")
    dis = dis[dis["model"] == "segformer"].dropna(subset=["abs_err_cd_mean"])
    dstar = json.load(open(ROOT / "output/revision_v4/e4d_routing.json"))["calibration"]["p95"]

    for split, color, label in (("standard", BASE_MID, "in-dist"),
                                ("extreme", RED_STRONG, "Extreme")):
        sub = dis[dis["split"] == split]
        ax.scatter(sub["disagreement"], sub["abs_err_cd_mean"].clip(lower=1e-3),
                   s=7, c=color, alpha=0.8, linewidths=0, label=label)
    ax.axvline(dstar, color="black", lw=1.0, ls="--")

    sub9 = dis[(dis["split"] == "extreme") & (dis["name"].map(norm_name) == SAMPLE)]
    ax.scatter(sub9["disagreement"], sub9["abs_err_cd_mean"], marker="*", s=85,
               c=RED_STRONG, edgecolors="black", linewidths=0.5, zorder=5)
    ax.annotate("sample (a)", (float(sub9["disagreement"].iloc[0]),
                               float(sub9["abs_err_cd_mean"].iloc[0])),
                textcoords="offset points", xytext=(-6, -9), fontsize=6,
                ha="right")

    ax.text(dstar * 1.13, 20, "$d^{\\star}$ (in-dist P95)", fontsize=6,
            ha="left", va="center")
    ax.text(0.975, 0.04,
            "AUROC 0.910\nroute flagged → HRNet:\nworst 48.2 → 3.4 px",
            transform=ax.transAxes, fontsize=6.2, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cross-frontend disagreement $d$", fontsize=7)
    ax.set_ylabel("SegFormer CD MAE (px)", fontsize=7)
    ax.set_title("(e) Reference-free guard + routing", fontsize=7.5,
                 fontweight="bold")
    ax.legend(fontsize=5.5, loc="center right", handletextpad=0.1,
              labelspacing=0.2)
    ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_loop(ax) -> None:
    """Design-manufacturing loop, marking the learned stage as the one that
    carries no run-time check."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 25)
    ax.axis("off")

    stages = [
        (2.0, "Layout\n(design intent)", BASE_MID),
        (21.0, "Litho\n+ SEM", BASE_MID),
        (40.0, "Segmentation\nfrontend", RED_STRONG),
        (59.0, "Metrology\nCD / LWR / LER", BASE_MID),
        (78.0, "Design decision\nPW · topology", BASE_MID),
    ]
    w, h, y = 18.0, 9.0, 12.5
    for x, label, color in stages:
        learned = color == RED_STRONG
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.1",
            linewidth=1.6 if learned else 0.9,
            edgecolor=color, facecolor="white" if learned else "#F4F6FB",
            zorder=3))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=6.2, color=color if learned else "#303030",
                fontweight="bold" if learned else "normal", zorder=4)

    for x, _, _ in stages[:-1]:
        ax.annotate("", xy=(x + w + 0.9, y + h / 2),
                    xytext=(x + w + 0.1, y + h / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color=NEUTRAL_MID))

    # Feedback edge as an explicit three-segment routing below the chain, so
    # it never crosses a box and its curvature does not depend on the renderer.
    x_from, x_to, y_bus = 78.0 + w / 2, 2.0 + w / 2, 5.2
    ax.plot([x_from, x_from, x_to], [y - 0.3, y_bus, y_bus],
            color=NEUTRAL_MID, lw=1.0, linestyle=(0, (4, 2)),
            solid_capstyle="butt", zorder=2)
    ax.annotate("", xy=(x_to, y - 0.3), xytext=(x_to, y_bus),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=NEUTRAL_MID,
                                linestyle=(0, (4, 2)), shrinkA=0, shrinkB=0))
    ax.text((x_from + x_to) / 2, y_bus - 1.0,
            "margin, model calibration, disposition",
            ha="center", va="top", fontsize=5.8, color=NEUTRAL_MID,
            style="italic")

    ax.text(40.0 + w / 2, y + h + 1.4,
            "trained model, no reference at run time",
            ha="center", va="bottom", fontsize=5.8, color=RED_STRONG)


def main() -> None:
    # Row (a) is width-driven: the four square tiles need ~1.57 in, so the row
    # is sized just above that and the scatter row takes what is left.
    fig = plt.figure(figsize=(7.16, 3.94))
    gs = gridspec.GridSpec(2, 12, figure=fig, height_ratios=[1.32, 1.0],
                           hspace=0.34, wspace=2.2,
                           left=0.075, right=0.985, top=0.935, bottom=0.105)
    panel_a(fig, gs[0, :])
    fig.text(0.075, 0.962,
             "(a) One out-of-window sample, and the equal-IoU ambiguity "
             "that defeats overlap in principle",
             fontsize=7.5, fontweight="bold")

    panel_b(fig.add_subplot(gs[1, 0:6]))
    panel_c(fig.add_subplot(gs[1, 6:12]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(f"{OUT}.{ext}", **kw)
        print(f"wrote {OUT}.{ext}")


if __name__ == "__main__":
    main()
