"""统一 baseline 评估入口.

用法:
    python scripts/baselines/eval_baseline.py \
        --config scripts/baselines/configs/unet.yaml \
        --ckpt output/baselines/unet/best.pth \
        --data-root dataset/litho \
        --split test \
        --output output/baselines/unet/eval_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.baselines._cfg import load_config
from specedge.baselines import build_model
from specedge.data.litho_dataset import LithoSegDataset
from specedge.metrics_psd import evaluate_pair


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--output", default=None, help="结果 JSON 输出路径; 默认 ckpt 同目录 eval_<split>.json")
    p.add_argument("--save-preds", default=None, help="可选: 保存 pred mask / overlay / error map 的目录")
    p.add_argument("--num-workers", type=int, default=None, help="覆盖 cfg.data.num_workers; 调试/本机评估可设为 0")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def logits_to_mask(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        return (torch.sigmoid(logits) > 0.5).long().squeeze(1)
    return logits.argmax(dim=1)


def boundary(mask: np.ndarray) -> np.ndarray:
    eroded = ndi.binary_erosion(mask.astype(bool))
    return np.logical_and(mask.astype(bool), np.logical_not(eroded))


def save_prediction_artifacts(
    save_root: Path,
    name: str,
    image: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
) -> None:
    """Save mask and compact visual diagnostics at evaluation resolution."""
    mask_dir = save_root / "masks"
    overlay_dir = save_root / "overlays"
    error_dir = save_root / "errors"
    boundary_dir = save_root / "boundaries"
    for d in (mask_dir, overlay_dir, error_dir, boundary_dir):
        d.mkdir(parents=True, exist_ok=True)

    img_u8 = np.clip(image.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    pred_u8 = (pred.astype(np.uint8) * 255)
    Image.fromarray(pred_u8).save(mask_dir / f"{name}.png")

    pb = boundary(pred)
    gb = boundary(gt)

    overlay = img_u8.copy()
    overlay[gb] = np.array([0, 220, 80], dtype=np.uint8)      # GT: green
    overlay[pb] = np.array([255, 60, 60], dtype=np.uint8)     # pred: red
    overlay[np.logical_and(pb, gb)] = np.array([255, 230, 0], dtype=np.uint8)
    Image.fromarray(overlay).save(overlay_dir / f"{name}.png")

    err = np.zeros_like(img_u8)
    tp = np.logical_and(pred == 1, gt == 1)
    fp = np.logical_and(pred == 1, gt == 0)
    fn = np.logical_and(pred == 0, gt == 1)
    err[tp] = np.array([210, 210, 210], dtype=np.uint8)
    err[fp] = np.array([255, 80, 80], dtype=np.uint8)
    err[fn] = np.array([70, 130, 255], dtype=np.uint8)
    Image.fromarray(err).save(error_dir / f"{name}.png")

    bd = img_u8.copy()
    bd[gb] = np.array([0, 220, 80], dtype=np.uint8)
    bd[pb] = np.array([255, 60, 60], dtype=np.uint8)
    Image.fromarray(bd).save(boundary_dir / f"{name}.png")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root

    img_size = tuple(cfg["data"]["image_size"])
    dataset = LithoSegDataset(cfg["data"]["root"], args.split, img_size, transform=None)
    loader = DataLoader(
        dataset,
        batch_size=max(1, cfg["data"]["batch_size"] // 2),
        shuffle=False,
        num_workers=cfg["data"]["num_workers"] if args.num_workers is None else args.num_workers,
    )

    model = build_model(cfg["model"], num_classes=cfg["data"]["num_classes"]).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    eval_cfg = cfg["eval"]
    per_sample: list[dict] = []
    times: list[float] = []
    save_root = Path(args.save_preds) if args.save_preds else None

    for batch in loader:
        img = batch["image"].to(args.device, non_blocking=True)
        mask = batch["mask"].to(args.device, non_blocking=True)
        names = batch["name"]

        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        logits = model(img)
        if args.device == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) / img.shape[0])

        pred = logits_to_mask(logits).cpu().numpy().astype(np.uint8)
        gt = mask.squeeze(1).long().cpu().numpy().astype(np.uint8)
        img_cpu = batch["image"].cpu().numpy()
        for i in range(pred.shape[0]):
            metrics = evaluate_pair(
                pred[i],
                gt[i],
                boundary_tolerance_px=eval_cfg.get("boundary_tolerance_px", 2),
                hf_cutoff_ratio=eval_cfg.get("psd", {}).get("hf_cutoff_ratio", 0.5),
            )
            metrics["name"] = names[i]
            per_sample.append(metrics)
            if save_root is not None:
                save_prediction_artifacts(save_root, names[i], img_cpu[i], pred[i], gt[i])

    keys = [k for k in per_sample[0].keys() if k != "name"]
    summary = {}
    for k in keys:
        vals = [s[k] for s in per_sample if np.isfinite(s[k])]
        summary[k] = float(np.mean(vals)) if vals else float("nan")
    summary["inference_s_per_image"] = float(np.mean(times))
    summary["num_samples"] = len(per_sample)

    out_path = Path(args.output) if args.output else Path(args.ckpt).parent / f"eval_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_sample": per_sample}, f, indent=2)

    print(f"[{args.split}] " + ", ".join(f"{k}={v:.4f}" for k, v in summary.items() if isinstance(v, float)))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
