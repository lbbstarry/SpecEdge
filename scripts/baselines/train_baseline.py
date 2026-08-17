"""统一 baseline 训练入口.

用法:
    python scripts/baselines/train_baseline.py \
        --config scripts/baselines/configs/unet.yaml \
        --data-root dataset/litho \
        --output output/baselines/unet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.baselines._cfg import load_config
from specedge.baselines import build_model
from specedge.data.litho_dataset import LithoSegDataset, SimpleAugment
from specedge.metrics_psd import evaluate_pair


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data-root", default=None, help="覆盖 cfg.data.root")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    img_size = tuple(data_cfg["image_size"])
    aug_cfg = data_cfg.get("augment", {})
    train_aug = SimpleAugment(**aug_cfg)

    train_set = LithoSegDataset(data_cfg["root"], "train", img_size, transform=train_aug)
    val_set = LithoSegDataset(data_cfg["root"], "val", img_size, transform=None)
    train_loader = DataLoader(
        train_set,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, data_cfg["batch_size"] // 2),
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """target shape (N, 1, H, W) in {0,1}. logits 单通道或双通道均支持."""
    if logits.shape[1] == 1:
        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target)
    else:
        prob = F.softmax(logits, dim=1)[:, 1:2]
        bce = F.cross_entropy(logits, target.squeeze(1).long())
    inter = (prob * target).sum(dim=(2, 3))
    denom = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = 1.0 - (2 * inter + eps) / (denom + eps)
    return bce + dice.mean()


def logits_to_mask(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        return (torch.sigmoid(logits) > 0.5).long().squeeze(1)
    return logits.argmax(dim=1)


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: str, eval_cfg: dict) -> dict[str, float]:
    model.eval()
    agg: dict[str, list[float]] = {}
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits = model(img)
        pred = logits_to_mask(logits).cpu().numpy()
        gt = mask.squeeze(1).long().cpu().numpy()
        for i in range(pred.shape[0]):
            metrics = evaluate_pair(
                pred[i].astype(np.uint8),
                gt[i].astype(np.uint8),
                boundary_tolerance_px=eval_cfg.get("boundary_tolerance_px", 2),
                hf_cutoff_ratio=eval_cfg.get("psd", {}).get("hf_cutoff_ratio", 0.5),
            )
            for k, v in metrics.items():
                if np.isfinite(v):
                    agg.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in agg.items()}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.resolved.json", "w") as f:
        json.dump(cfg, f, indent=2)

    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    train_loader, val_loader = make_loaders(cfg)
    model = build_model(cfg["model"], num_classes=cfg["data"]["num_classes"]).to(args.device)

    train_cfg = cfg["train"]
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=train_cfg["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("amp", False) and args.device == "cuda")

    best_iou = -1.0
    log_path = out_dir / "train.log.jsonl"
    log_f = open(log_path, "w")

    for epoch in range(train_cfg["epochs"]):
        model.train()
        t0 = time.time()
        running = 0.0
        for step, batch in enumerate(train_loader):
            img = batch["image"].to(args.device, non_blocking=True)
            mask = batch["mask"].to(args.device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(img)
                # 二值场景把 mask 用于双通道也 ok: cross_entropy 内部会处理
                if logits.shape[1] == 2:
                    target = mask
                else:
                    target = mask
                loss = bce_dice_loss(logits, target)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += loss.item()

        sched.step()
        train_loss = running / max(1, len(train_loader))

        val_metrics = validate(model, val_loader, args.device, cfg["eval"])
        rec = {"epoch": epoch, "train_loss": train_loss, "time_s": time.time() - t0, **val_metrics}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(f"[ep {epoch:03d}] loss={train_loss:.4f} iou={val_metrics.get('iou', 0):.4f} "
              f"bf1={val_metrics.get('boundary_f1', 0):.4f} "
              f"hf_ratio_pred={val_metrics.get('edge_psd_hf_ratio_pred', 0):.4f}")

        if val_metrics.get("iou", 0) > best_iou:
            best_iou = val_metrics["iou"]
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, out_dir / "best.pth")

    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": train_cfg["epochs"] - 1}, out_dir / "last.pth")
    log_f.close()
    print(f"done. best val IoU = {best_iou:.4f}. ckpt -> {out_dir/'best.pth'}")


if __name__ == "__main__":
    main()
