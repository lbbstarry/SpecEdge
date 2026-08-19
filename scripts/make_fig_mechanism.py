"""Figure: controlled test of the overlap/measurand decoupling mechanism.

Reads the E23 summary (586 reference masks perturbed by displacement fields of
equal L1 norm and different shape) and plots the two halves of the argument
side by side, sharing the x axis:

  left  -- CD MAE against IoU. Overlap reads the L1 norm, so the three fields
           land at one IoU; CD reads a signed mean, so it spans ~26x.
  right -- LER 3sigma against IoU. The constant offset is removed exactly by
           detrending and sits on the reference level; the two zero-mean
           fields raise it.

Run manually, like the other make_fig*.py scripts:
    python scripts/make_fig_mechanism.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output/revision_v4/e23_mechanism/summary.json"
OUT = ROOT / "paper/figures/fig_mechanism"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman",
                              "Liberation Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 0.6
plt.rcParams["legend.frameon"] = False

# One marker per displacement field; colours match the paper's callout palette.
ARMS = [
    ("constant", r"constant $+c$", "o", "#C0392B"),
    ("rademacher", r"two-point $\pm c$", "s", "#2E6DA4"),
    ("gaussian", r"Gaussian, $\overline{|\delta|}=c$", "^", "#1E8449"),
]


def main() -> None:
    d = json.loads(SRC.read_text())
    by_c = d["by_c"]
    cs = sorted(by_c, key=float)
    ler_ref = by_c[cs[0]]["arms"]["constant"]["ler_ref_mean"]

    # Each amplitude c yields one IoU for all three fields; tick there so the
    # tie bars sit on the axis and the "same overlap" reading is unambiguous.
    ticks = [sum(by_c[c]["arms"][n]["iou_mean"] for n, _, _, _ in ARMS) / len(ARMS)
             for c in cs]

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(3.45, 1.48),
        gridspec_kw=dict(left=0.135, right=0.995, top=0.985, bottom=0.375,
                         wspace=0.50))

    for ax, key, ylabel in ((axl, "cd_err_mean", "CD MAE (px)"),
                            (axr, "ler_pert_mean", r"LER $3\sigma$ (px)")):
        for x, c in zip(ticks, cs):  # one amplitude, hence one overlap
            ys = [by_c[c]["arms"][name][key] for name, _, _, _ in ARMS]
            ax.plot([x, x], [min(ys), max(ys)], color="0.65", lw=0.7, zorder=1)
        for name, label, marker, colour in ARMS:
            ax.plot(ticks, [by_c[c]["arms"][name][key] for c in cs],
                    marker=marker, ms=3.2, lw=0.7, color=colour,
                    label=label, zorder=3)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.3f}\n$c\\!=\\!{float(c):.0f}$"
                            for t, c in zip(ticks, cs)])
        ax.set_xlabel("IoU", fontsize=6.5, labelpad=1.0)
        ax.set_ylabel(ylabel, fontsize=6.5, labelpad=1.5)
        ax.tick_params(labelsize=5.6, length=2, pad=1.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.invert_xaxis()  # larger perturbation to the right

    axl.set_yscale("log")
    axr.axhline(ler_ref, color="0.35", lw=0.7, ls=(0, (3, 2)), zorder=2)
    axr.text(0.04, ler_ref, "reference level",
             transform=axr.get_yaxis_transform(),
             fontsize=5.2, color="0.35", ha="left", va="bottom")

    # the headline CD contrast, at the largest amplitude
    top = by_c[cs[-1]]["arms"]
    axl.annotate(r"$26\times$",
                 xy=(ticks[-1], (top["constant"]["cd_err_mean"]
                                 * top["rademacher"]["cd_err_mean"]) ** 0.5),
                 xytext=(3, 0), textcoords="offset points",
                 fontsize=6, color="#C0392B", ha="left", va="center")

    fig.legend(handles=axl.lines[3:6] if len(axl.lines) >= 6 else None,
               loc="lower center", bbox_to_anchor=(0.55, -0.015), ncol=3,
               fontsize=5.5, handletextpad=0.3, columnspacing=0.8,
               handlelength=1.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", **kw)
        print(f"wrote {OUT}.{ext}")


if __name__ == "__main__":
    main()
