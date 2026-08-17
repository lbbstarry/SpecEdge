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

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
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

    cells = [
        (None, "SEM input", "out-of-window · fg 0.11", "white"),
        (otsu, "Threshold (industry)", "misses the line · IoU 0.14", "#FFB4AE"),
        (segf, "SegFormer · IoU 0.986", "hallucination · CD err 48.2 px", "#FFB4AE"),
        (hrnet, "HRNet · IoU 0.984", "accurate · CD err 0.07 px", "#A8E6B8"),
    ]
    axes = []
    inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_hero,
                                             wspace=0.045)
    for i, (pred, label, verdict, vc) in enumerate(cells):
        ax = fig.add_subplot(inner[i])
        plate_cell(ax, sem, pred, gt, label, verdict, vc)
        axes.append(ax)

    # scale bar on the first cell (pixel units; nm calibration pending)
    axes[0].plot([1024 - 60 - 200, 1024 - 60], [1024 - 55, 1024 - 55],
                 color="white", lw=1.6)
    axes[0].text(1024 - 60 - 100, 1024 - 85, "200 px", color="white",
                 fontsize=5.5, ha="center", va="bottom")

    # shared overlay legend under the strip
    handles = [
        plt.Line2D([], [], marker="s", ls="", mfc=RED_CALLOUT, mec="none",
                   ms=5, label="spurious foreground (pred ∖ ref)"),
        plt.Line2D([], [], marker="s", ls="", mfc=ORANGE, mec="none",
                   ms=5, label="missed foreground (ref ∖ pred)"),
        plt.Line2D([], [], color=CYAN, lw=1.4, label="reference boundary"),
    ]
    axes[0].legend(handles=handles, loc="upper left",
                   bbox_to_anchor=(0.0, -0.015), fontsize=6, ncol=3,
                   handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0)


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

    n_clip = int((df["iou"] < 0.965).sum())
    for model in MODEL_COLORS:  # family order; SegFormer drawn last, on top
        sub = df[df["model"] == model]
        ax.scatter(sub["iou"], sub["cd"], s=6, c=MODEL_COLORS[model],
                   label=MODEL_LABELS[model], alpha=0.8, linewidths=0)
    ax.axvspan(0.99, 0.995, color=BG_AQUA, zorder=0)
    binned = df[(df["iou"] >= 0.99) & (df["iou"] < 0.995)]["cd"]
    p5, p95 = np.percentile(binned, [5, 95])
    ax.annotate("", xy=(0.9925, p95), xytext=(0.9925, p5),
                arrowprops=dict(arrowstyle="<->", color="black", lw=0.9))
    ax.text(0.9887, np.sqrt(p5 * p95), f"{p95 / p5:.0f}× spread",
            fontsize=7, fontweight="bold", ha="right", va="center")
    ax.set_xlim(0.965, 0.9975)
    ax.text(0.02, 0.02, f"$\\leftarrow$ {n_clip} samples, IoU < 0.965",
            transform=ax.transAxes, fontsize=5.5, color=NEUTRAL_MID,
            va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("per-sample IoU", fontsize=7)
    ax.set_ylabel("CD MAE (px)", fontsize=7)
    ax.set_title("(b) IoU does not rank metrology", fontsize=7.5,
                 fontweight="bold")
    ax.legend(fontsize=5.5, loc="upper left", handletextpad=0.1,
              borderaxespad=0.2, labelspacing=0.2)
    ax.tick_params(labelsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_c(ax) -> None:
    onset = json.load(open(ROOT / "output/revision_v4/e3_onset.json"))
    tau, bp, ci = onset["tau_px"], onset["breakpoint"], onset["breakpoint_ci95"]

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
    ax.text(bp + 0.008, 2e-3, f"breakpoint {bp:.2f}", fontsize=6, color=ORANGE,
            rotation=90, va="bottom")
    ax.text(0.745, tau * 1.5, "$\\tau_\\sigma$ = 2.65 px", fontsize=6,
            color=NEUTRAL_MID, ha="right")
    ax.set_yscale("log")
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
    fig = plt.figure(figsize=(7.16, 4.7))
    gs = gridspec.GridSpec(2, 12, figure=fig, height_ratios=[1.32, 1.0],
                           hspace=0.44, wspace=2.2,
                           left=0.075, right=0.985, top=0.925, bottom=0.085)
    panel_a(fig, gs[0, :])
    fig.text(0.075, 0.962,
             "(a) Two frontends that are equivalent in distribution, on one "
             "out-of-window sample",
             fontsize=7.5, fontweight="bold")

    panel_b(fig.add_subplot(gs[1, 0:6]))
    panel_c(fig.add_subplot(gs[1, 6:12]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(f"{OUT}.{ext}", **kw)
        print(f"wrote {OUT}.{ext}")


if __name__ == "__main__":
    main()
