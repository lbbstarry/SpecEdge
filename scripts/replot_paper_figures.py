"""Re-render the five paper figures in a unified IEEE-style layout.

Reads cached CSV/JSON from output/revision_v4 (no recomputation, no GPU),
writes into paper/figures/ replacing the analysis-style PNGs the LaTeX
already references. English labels, Times-like serif, panel letters, common
color palette, 300 DPI.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RV = REPO / "output/revision_v4"
HARD = REPO / "output/hard_eval"
STD_BASE = REPO / "output/baselines"
STD_METRO = REPO / "output/metrology"
PFIG = REPO / "paper/figures"
PFIG.mkdir(parents=True, exist_ok=True)

PAL = {
    "unet": "#4878d0",
    "deeplabv3plus": "#ee854a",
    "hrnet": "#6acc64",
    "segformer": "#d65f5f",
    "in_dist": "#4878d0",
    "extreme": "#d65f5f",
    "band": "#f4a261",
    "ref": "#888888",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODELS = ["unet", "deeplabv3plus", "hrnet", "segformer"]


def load_metro(split: str, model: str) -> pd.DataFrame:
    p = STD_METRO / f"{model}_test_metrics.csv" if split == "standard" else HARD / f"{model}_metrology.csv"
    return pd.read_csv(p, dtype={"name": str})


def load_iou(split: str, model: str) -> pd.DataFrame:
    p = STD_BASE / f"{model}/eval_test.json" if split == "standard" else HARD / f"{model}_eval.json"
    d = json.load(open(p))
    return pd.DataFrame([{"name": str(s["name"]), "iou": s["iou"]} for s in d["per_sample"]])


def panel_label(ax, text):
    ax.text(-0.18, 1.05, text, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")


# ---------- Fig E3: onset region -------------------------------------------
def fig_onset() -> None:
    onset = json.load(open(RV / "e3_onset.json"))
    sf_std = load_metro("standard", "segformer").dropna(subset=["abs_err_cd_mean", "gt_foreground_ratio"])
    sf_ext = load_metro("extreme", "segformer").dropna(subset=["abs_err_cd_mean", "gt_foreground_ratio"])
    lo, hi = onset["breakpoint_ci95"]
    bp = onset["breakpoint"]
    tau = onset["tau_px"]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.axvspan(lo, hi, color=PAL["band"], alpha=0.30,
               label=f"breakpoint 95% CI [{lo:.2f}, {hi:.2f}]")
    ax.axvline(bp, color=PAL["band"], lw=0.8, ls="-", alpha=0.8)
    ax.axhline(tau, ls="--", color="k", lw=0.7,
               label=f"$\\tau_\\sigma = {tau:.2f}$ px")
    ax.scatter(sf_std["gt_foreground_ratio"], sf_std["abs_err_cd_mean"] + 1e-3,
               s=14, alpha=0.65, color=PAL["in_dist"],
               edgecolors="none", label="In-dist test")
    ax.scatter(sf_ext["gt_foreground_ratio"], sf_ext["abs_err_cd_mean"] + 1e-3,
               s=18, alpha=0.85, color=PAL["extreme"],
               edgecolors="none", label="Extreme")
    ax.set_yscale("log")
    ax.set_xlabel("GT foreground ratio")
    ax.set_ylabel("SegFormer per-sample CD MAE (px)")
    ax.set_title("SegFormer failure onset (Extreme-only changepoint)")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(PFIG / "fig_e3_onset.png")
    plt.close(fig)


# ---------- Fig E4: disagreement guard -------------------------------------
def fig_guard() -> None:
    df = pd.read_csv(RV / "e4_disagreement.csv")
    summary = json.load(open(RV / "e4_summary.json"))
    onset = json.load(open(RV / "e3_onset.json"))
    tau = onset["tau_px"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    sub_sf = df[df.model == "segformer"]
    for split, color in [("standard", PAL["in_dist"]), ("extreme", PAL["extreme"])]:
        sel = sub_sf[sub_sf.split == split]
        axes[0].scatter(sel["disagreement"], sel["abs_err_cd_mean"] + 1e-3,
                        s=18, alpha=0.75, color=color, edgecolors="none",
                        label=f"{'In-dist' if split=='standard' else 'Extreme'}")
    axes[0].axhline(tau, ls="--", color="k", lw=0.7,
                    label=f"$\\tau_\\sigma$")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Disagreement $d$ (1 - mean IoU vs others)")
    axes[0].set_ylabel("SegFormer per-sample CD MAE (px)")
    axes[0].set_title("(a) Disagreement vs. SegFormer CD MAE")
    axes[0].legend(loc="upper left", frameon=False)
    panel_label(axes[0], "(a)")

    sub = df[df.split == "extreme"]
    lab = sub["abs_err_cd_mean"].to_numpy(float) > tau
    sc = sub["disagreement"].to_numpy(float)
    order = np.argsort(-sc)
    sc_s, lab_s = sc[order], lab[order]
    tpr = np.cumsum(lab_s) / max(lab_s.sum(), 1)
    fpr = np.cumsum(~lab_s) / max((~lab_s).sum(), 1)
    auroc = summary["extreme_all_models"]["auroc_disagreement"]
    axes[1].plot([0, 1], [0, 1], "--", color="k", lw=0.5)
    axes[1].plot(fpr, tpr, "-", lw=1.5, color=PAL["extreme"],
                 label=f"Disagreement, AUROC = {auroc:.3f}")
    fg_dev = summary["extreme_all_models"]["auroc_fg_dev"]
    cc_dev = summary["extreme_all_models"]["auroc_cc_dev"]
    axes[1].plot([], [], " ",
                 label=f"single-model baselines: fg dev {fg_dev:.3f}, cc dev {cc_dev:.3f}")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("(b) ROC on Extreme, all frontends pooled")
    axes[1].legend(loc="lower right", frameon=False)
    panel_label(axes[1], "(b)")
    fig.tight_layout()
    fig.savefig(PFIG / "fig_e4_guard.pdf")  # vector: the version the paper includes
    fig.savefig(PFIG / "fig_e4_guard.png")
    plt.close(fig)


# ---------- Fig E5b: training support --------------------------------------
def fig_train_support() -> None:
    tr = json.load(open(RV / "e5b_train_support.json"))
    train_min = tr["train_fg_min"]

    from PIL import Image
    train_fgs = []
    for p in sorted((REPO / "dataset/litho/masks/train").glob("*.png")):
        m = np.asarray(Image.open(p).convert("L"))
        train_fgs.append(float((m > 127).mean()))
    train_fgs = np.asarray(train_fgs)
    std = load_metro("standard", "segformer")["gt_foreground_ratio"].dropna().to_numpy(float)
    ext = load_metro("extreme", "segformer")["gt_foreground_ratio"].dropna().to_numpy(float)

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bins = np.linspace(0, 0.9, 36)
    ax.hist(train_fgs, bins=bins, alpha=0.55, color="#9ecae1",
            label=f"train (n={len(train_fgs)})", density=True)
    ax.hist(std, bins=bins, alpha=0.55, color=PAL["in_dist"],
            label=f"In-dist (n={len(std)})", density=True)
    ax.hist(ext, bins=bins, alpha=0.55, color=PAL["extreme"],
            label=f"Extreme (n={len(ext)})", density=True)
    ax.axvline(train_min, ls="--", color="k", lw=0.8,
               label=f"train fg min = {train_min:.3f}")
    ax.set_xlabel("GT foreground ratio")
    ax.set_ylabel("Density")
    ax.set_title("Training support across splits")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(PFIG / "fig_e5b_train_support.pdf")  # vector: the version the paper includes
    fig.savefig(PFIG / "fig_e5b_train_support.png")
    plt.close(fig)


# ---------- Fig E7: IoU vs CD MAE conditional spread ------------------------
def fig_iou_vs_cd() -> None:
    frames = []
    for m in MODELS:
        d = load_iou("standard", m).merge(load_metro("standard", m), on="name")
        d["model"] = m
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for m in MODELS:
        sel = df[df.model == m]
        ax.scatter(sel["iou"], sel["abs_err_cd_mean"] + 1e-3,
                   s=14, alpha=0.7, color=PAL[m],
                   edgecolors="none", label={
                       "unet": "U-Net", "deeplabv3plus": "DeepLabV3+",
                       "hrnet": "HRNet", "segformer": "SegFormer"}[m])
    for x in [0.98, 0.985, 0.99, 0.995]:
        ax.axvline(x, color="grey", lw=0.3, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlim(0.97, 1.001)
    ax.set_xlabel("per-sample IoU")
    ax.set_ylabel("per-sample CD MAE (px)")
    ax.set_title("Decoupling: CD MAE inside narrow IoU bins")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(PFIG / "fig_e7_iou_vs_cd.png")
    plt.close(fig)


# ---------- Fig E8: r_bbox confound ----------------------------------------
def fig_rbbox() -> None:
    rb = pd.read_csv(RV / "e8_rbbox.csv")
    summary = json.load(open(RV / "summary_revision_v4.json"))["e8"]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    sc = ax.scatter(rb["r_bbox"], rb["abs_err_cd_mean"] + 1e-3,
                    c=rb["gt_foreground_ratio"], cmap="viridis", s=22,
                    edgecolors="white", linewidth=0.3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("GT foreground ratio", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_yscale("log")
    ax.set_xlabel(r"Layout-bbox coverage $r_{\mathrm{bbox}}$")
    ax.set_ylabel("SegFormer CD MAE (px)")
    pr = summary["partial_rbbox_vs_err_given_fg"]
    pv = summary["p_partial"]
    ax.set_title(rf"$r_\mathrm{{bbox}}$ confound: partial $r$ = {pr:.2f}, $p$ = {pv:.2f}")
    fig.savefig(PFIG / "fig_e8_rbbox.pdf")  # vector: the version the paper includes
    fig.savefig(PFIG / "fig_e8_rbbox.png")
    plt.close(fig)


def main() -> None:
    fig_onset()
    fig_guard()
    fig_train_support()
    fig_iou_vs_cd()
    fig_rbbox()
    print(f"5 figures regenerated under {PFIG}")


if __name__ == "__main__":
    main()
