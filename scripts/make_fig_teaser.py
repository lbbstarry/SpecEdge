"""Page-1 teaser: a directly predicted mask is not measurement-grade.

Three tiles cropped from Fig. 1 (Stage 2 row) of the authors' own LithoSeg
manuscript (docs/DATE__LithoSeg__final_.pdf, arXiv:2511.12005; embedded
285-ppi raster extracted with pdfimages, tile frames trimmed; crops stored
under paper/figures/src_lithoseg/):

  full     -- SAM, box-prompted, on one advanced-node SEM field; blue mask,
              red box marks the zoom window
  direct   -- the boxed window: the mask boundary sits inside the resist
              edge; green bars are the normals along which 1-D intensity
              profiles are read
  refined  -- same window after LithoSeg's 1-D profile regression: the red
              contour lies on the edge

The colors are the source figure's own (blue mask / red refined contour /
green normals) and are named in the caption; the overview figure's
claim-grammar legend (cyan/red) starts one figure later.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
SRC = ROOT / "paper" / "figures" / "src_lithoseg"
OUT = ROOT / "paper" / "figures" / "fig_teaser"

TILES = [
    ("full", "SAM mask, direct", None),
    ("direct", "boxed window, zoomed", "boundary off the edge"),
    ("refined", "after 1-D refinement", "boundary on the edge"),
]


def main() -> None:
    fig = plt.figure(figsize=(3.5, 1.42))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.045,
                           left=0.005, right=0.995, top=0.845, bottom=0.01)
    for ci, (name, title, note) in enumerate(TILES):
        ax = fig.add_subplot(gs[0, ci])
        ax.imshow(np.asarray(Image.open(SRC / f"{name}.png")),
                  interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=6.5, fontweight="bold", pad=3)
        if note is not None:
            ax.text(0.5, 0.045, note, transform=ax.transAxes, fontsize=5.4,
                    color="white", ha="center", va="bottom")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {"dpi": 600}), ("png", {"dpi": 300})):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", **kw)
        print(f"wrote {OUT}.{ext}")


if __name__ == "__main__":
    main()
