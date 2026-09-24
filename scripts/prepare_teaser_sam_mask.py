"""Produce the direct-segmentation mask shown in the page-1 teaser figure.

Runs box-prompted SAM (ViT-H) on one SEM field of the study dataset and
saves the union of the per-line masks. Each line gets one box prompt (the
bounding box of the corresponding reference-mask component, padded 5 px)
plus one positive point at the component centroid; of the three candidate
masks SAM returns, the one with the highest predicted IoU is kept. The
reference mask supplies prompt geometry only; the saved mask is SAM's own
output. make_fig_teaser.py consumes the result.

Usage:
  python scripts/prepare_teaser_sam_mask.py \
      --checkpoint /path/to/sam_vit_h_4b8939.pth [--sam-code /path/to/pkg]
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "dataset" / "SEM" / "ADI_train"
REF = ROOT / "dataset" / "SEM" / "train_new_gt"
OUT = ROOT / "output" / "teaser"

SAMPLE = "00000040"
MIN_COMPONENT_PX = 500
BOX_PAD_PX = 5


def load_sam(checkpoint: str, sam_code: str | None):
    if sam_code:
        sys.path.insert(0, sam_code)
    from segment_anything import sam_model_registry, SamPredictor

    sam = sam_model_registry["vit_h"](checkpoint=checkpoint).to("cuda")
    # Some vendored SAM variants extend PromptEncoder.forward with an extra
    # text_embeds argument without a default; feed None so the stock
    # box/point path still works.
    sig = inspect.signature(sam.prompt_encoder.forward)
    extra = sig.parameters.get("text_embeds")
    if extra is not None and extra.default is inspect.Parameter.empty:
        orig = sam.prompt_encoder.forward
        sam.prompt_encoder.forward = (
            lambda points, boxes, masks, text_embeds=None: orig(
                points, boxes, masks, text_embeds
            )
        )
    return SamPredictor(sam)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="SAM ViT-H .pth file")
    ap.add_argument("--sam-code", default=None,
                    help="optional path to prepend to sys.path so that "
                         "segment_anything can be imported")
    ap.add_argument("--sample", default=SAMPLE)
    args = ap.parse_args()

    image = np.asarray(Image.open(IMAGE / f"{args.sample}.bmp").convert("RGB"))
    ref = np.asarray(Image.open(REF / f"{args.sample}.png").convert("L")) > 127
    side = image.shape[0] - 1

    predictor = load_sam(args.checkpoint, args.sam_code)
    predictor.set_image(image)

    labels, n = ndimage.label(ref)
    union = np.zeros(ref.shape, bool)
    for k in range(1, n + 1):
        ys, xs = np.where(labels == k)
        if len(ys) < MIN_COMPONENT_PX:
            continue
        box = np.array([xs.min() - BOX_PAD_PX, ys.min() - BOX_PAD_PX,
                        xs.max() + BOX_PAD_PX, ys.max() + BOX_PAD_PX]).clip(0, side)
        centroid = np.array([[int(xs.mean()), int(ys.mean())]])
        masks, scores, _ = predictor.predict(
            box=box[None, :], point_coords=centroid,
            point_labels=np.array([1]), multimask_output=True)
        union |= masks[np.argmax(scores)]

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"sam_direct_{args.sample}.png"
    Image.fromarray((union * 255).astype(np.uint8)).save(out)
    inter = (union & ref).sum()
    iou = inter / (union | ref).sum()
    print(f"wrote {out}  (IoU vs reference {iou:.3f})")


if __name__ == "__main__":
    main()
