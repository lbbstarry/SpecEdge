"""E5a mechanism probe: SegFormer inference-mode intervention on Extreme worst cases.

The model was trained at 512x512. Two clean contrasts on worst-10 + best-5:

  resize512  1024 -> 512 bilinear, single forward (matches the cached baseline)
  tile512    full 1024 native, overlapping 512x512 tiles, stride 256, softmax averaged

Both feed 512x512 tensors to the network, so the position-embedding regime is
identical. The only difference is *which* 512 region the model sees: a globally
down-sampled view (resize512) vs. local high-resolution patches (tile512). If
the failure disappears under tile512, scale/global-context is the mechanism and
tile inference is a free deployment fix; if not, the architectural prior is the
mechanism and tile inference is not enough.

Per (sample, mode): pred-gt foreground delta, spurious component count
(extra components vs reference, area >= 64 px), CD MAE vs reference.
Outputs: output/revision_v4/e5a_probe/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.baselines._cfg import load_config
from specedge.baselines import build_model
from specedge.metrology import evaluate_metrology_pair

IMG_DIR = REPO / "dataset/litho_hard/images/hard"
REF_DIR = REPO / "dataset/litho_hard/masks/hard"
CKPT = REPO / "output/baselines/segformer/best.pth"
CFG = REPO / "scripts/baselines/configs/segformer.yaml"
OUT = REPO / "output/revision_v4/e5a_probe"
WORST10 = ["9", "10", "16", "14", "2", "7", "1", "22", "33", "51"]
BEST5 = ["34", "71", "66", "70", "36"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> torch.nn.Module:
    cfg = load_config(str(CFG))
    model = build_model(cfg["model"], num_classes=cfg["data"]["num_classes"])
    state = torch.load(CKPT, map_location="cpu")
    for key in ("model", "state_dict", "model_state"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return t.to(DEVICE)


@torch.no_grad()
def predict(model: torch.nn.Module, img_full: np.ndarray, mode: str) -> np.ndarray:
    h, w = img_full.shape[:2]
    if mode == "resize512":
        small = np.asarray(Image.fromarray(img_full).resize((512, 512), Image.BILINEAR))
        logits = model(to_tensor(small))
        mask = logits.argmax(1).squeeze(0).byte().cpu().numpy()
        return np.asarray(Image.fromarray(mask * 255).resize((w, h), Image.NEAREST)) > 127
    if mode == "tile512":
        acc = torch.zeros((1, 2, h, w), device=DEVICE)
        cnt = torch.zeros((1, 1, h, w), device=DEVICE)
        ts, stride = 512, 256
        for y0 in range(0, h - ts + 1, stride):
            for x0 in range(0, w - ts + 1, stride):
                tile = img_full[y0:y0 + ts, x0:x0 + ts]
                logits = model(to_tensor(tile))
                acc[:, :, y0:y0 + ts, x0:x0 + ts] += F.softmax(logits, dim=1)
                cnt[:, :, y0:y0 + ts, x0:x0 + ts] += 1
        return (acc / cnt).argmax(1).squeeze(0).cpu().numpy() > 0
    raise ValueError(mode)


def spurious_components(pred: np.ndarray, ref: np.ndarray, min_area: int = 64) -> int:
    extra = ((pred > 0) & (ref == 0)).astype(np.uint8)
    ncc, _, stats, _ = cv2.connectedComponentsWithStats(extra, 8)
    return int(sum(1 for j in range(1, ncc) if stats[j][4] >= min_area))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model()
    rows = []
    for group, names in [("worst10", WORST10), ("best5", BEST5)]:
        for name in names:
            img = np.asarray(Image.open(IMG_DIR / f"{name}.png").convert("RGB"))
            ref = (np.asarray(Image.open(REF_DIR / f"{name}.png").convert("L")) > 127).astype(np.uint8)
            for mode in ("resize512", "tile512"):
                pred = predict(model, img, mode).astype(np.uint8)
                Image.fromarray(pred * 255).save(OUT / f"{name}_{mode}.png")
                pair = evaluate_metrology_pair(pred * 255, ref * 255)
                cd = pair.get("abs_err_cd_mean")
                rows.append({
                    "name": name, "group": group, "mode": mode,
                    "pred_fg_minus_gt": float(pred.mean() - ref.mean()),
                    "spurious_components": spurious_components(pred, ref),
                    "cd_mae": float(cd) if cd is not None and np.isfinite(float(cd)) else float("nan"),
                })
                print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "probe_results.csv", index=False)
    summary = {}
    for group in ("worst10", "best5"):
        for mode in ("resize512", "tile512"):
            sub = df[(df.group == group) & (df.mode == mode)]
            summary[f"{group}_{mode}"] = {
                "fg_excess_mean": float(sub["pred_fg_minus_gt"].mean()),
                "spurious_components_mean": float(sub["spurious_components"].mean()),
                "cd_mae_mean": float(np.nanmean(sub["cd_mae"])),
                "cd_mae_median": float(np.nanmedian(sub["cd_mae"])),
            }
    json.dump(summary, open(OUT / "e5a_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
