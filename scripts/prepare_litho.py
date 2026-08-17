"""光刻 SEM 数据预处理.

输入:
  dataset/SEM/ADI_train/*.bmp        (RGBA SEM 图像)
  dataset/SEM/train_new_gt/*.{png,jpg}  (二值 GT, 但有 jpg 压坏的)

输出 (对接 specedge.data.litho_dataset.LithoSegDataset):
  dataset/litho/
    images/{train,val,test}/<stem>.png   (RGB)
    masks/{train,val,test}/<stem>.png    ({0, 255} 二值)
    splits.json                            (划分清单, 便于复现)

处理逻辑:
  - 跳过 ADI 中没有对应 GT 的样本
  - RGBA -> RGB
  - GT 任何 >127 的像素视为前景, 统一二值化, 强制写为 PNG
    (jpg 被压坏的灰度值也被一并归一化)
  - 按 8:1:1 划分 train/val/test, 固定 seed
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src-images", default="dataset/SEM/ADI_train")
    p.add_argument("--src-masks", default="dataset/SEM/train_new_gt")
    p.add_argument("--dst", default="dataset/litho")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   help="train/val/test 比例")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bin-threshold", type=int, default=127,
                   help="GT 二值化阈值, >此值视为前景")
    return p.parse_args()


def index_dir(d: Path) -> dict[str, Path]:
    """stem -> path 索引, 同 stem 多扩展时优先 png."""
    out: dict[str, Path] = {}
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        stem = p.stem
        if stem not in out or p.suffix.lower() == ".png":
            out[stem] = p
    return out


def convert_image(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="PNG")


def convert_mask(src: Path, dst: Path, threshold: int) -> tuple[int, int]:
    """二值化保存. 返回 (前景像素数, 总像素数)."""
    m = Image.open(src).convert("L")
    arr = np.asarray(m, dtype=np.uint8)
    binary = (arr > threshold).astype(np.uint8) * 255
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary, mode="L").save(dst, format="PNG")
    return int((binary > 0).sum()), int(binary.size)


def split_stems(stems: list[str], ratios: list[float], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    shuffled = stems[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def main() -> None:
    args = parse_args()
    src_img_dir = Path(args.src_images)
    src_msk_dir = Path(args.src_masks)
    dst_root = Path(args.dst)

    img_idx = index_dir(src_img_dir)
    msk_idx = index_dir(src_msk_dir)

    paired = sorted(set(img_idx) & set(msk_idx))
    only_img = sorted(set(img_idx) - set(msk_idx))
    only_msk = sorted(set(msk_idx) - set(img_idx))
    print(f"[scan] images={len(img_idx)} masks={len(msk_idx)} paired={len(paired)}")
    if only_img:
        print(f"[skip] {len(only_img)} images without mask, e.g. {only_img[:5]}")
    if only_msk:
        print(f"[skip] {len(only_msk)} masks without image, e.g. {only_msk[:5]}")

    splits = split_stems(paired, args.ratios, args.seed)
    print(f"[split] train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    fg_total = px_total = 0
    converted_jpg_masks = 0
    for split, stems in splits.items():
        for stem in stems:
            convert_image(img_idx[stem], dst_root / "images" / split / f"{stem}.png")
            src_msk = msk_idx[stem]
            fg, total = convert_mask(src_msk, dst_root / "masks" / split / f"{stem}.png", args.bin_threshold)
            fg_total += fg
            px_total += total
            if src_msk.suffix.lower() in {".jpg", ".jpeg"}:
                converted_jpg_masks += 1

    fg_ratio = fg_total / max(1, px_total)
    print(f"[stats] foreground pixel ratio = {fg_ratio:.4f}")
    if converted_jpg_masks:
        print(f"[warn] {converted_jpg_masks} mask(s) were JPEG, binarized via threshold={args.bin_threshold} "
              "(check visually for one of these to confirm)")

    meta = {
        "src_images": str(src_img_dir),
        "src_masks": str(src_msk_dir),
        "ratios": args.ratios,
        "seed": args.seed,
        "bin_threshold": args.bin_threshold,
        "counts": {k: len(v) for k, v in splits.items()},
        "skipped_images_without_mask": only_img,
        "skipped_masks_without_image": only_msk,
        "foreground_ratio": fg_ratio,
        "splits": splits,
    }
    (dst_root).mkdir(parents=True, exist_ok=True)
    with open(dst_root / "splits.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] -> {dst_root}")


if __name__ == "__main__":
    main()
