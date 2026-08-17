"""光刻 SEM 二值分割数据集.

期望目录结构:
    root/
      images/{train,val,test}/*.png
      masks/{train,val,test}/*.png

mask 取值 {0, 255} 或 {0, 1} 均可, 内部统一二值化为 {0, 1}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class LithoSegDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: tuple[int, int] = (512, 512),
        transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform

        img_dir = self.root / "images" / split
        mask_dir = self.root / "masks" / split
        if not img_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"missing images/{split} or masks/{split} under {self.root}")

        self.samples: list[tuple[Path, Path]] = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            mask_path = mask_dir / f"{img_path.stem}.png"
            if not mask_path.is_file():
                # 允许 mask 与 image 同后缀名
                alt = mask_dir / img_path.name
                if alt.is_file():
                    mask_path = alt
                else:
                    continue
            self.samples.append((img_path, mask_path))

        if not self.samples:
            raise RuntimeError(f"no paired samples in {img_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, img_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
        img = Image.open(img_path).convert("RGB").resize(self.image_size, Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize(self.image_size, Image.NEAREST)
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
        return img_np, mask_np

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.samples[idx]
        img, mask = self._load(img_path, mask_path)

        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # HWC -> CHW
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        return {"image": img_t, "mask": mask_t, "name": img_path.stem}


class SimpleAugment:
    """轻量增广: 翻转 + 90 度旋转 + 亮度对比度抖动."""

    def __init__(
        self,
        hflip: bool = True,
        vflip: bool = True,
        rotate90: bool = True,
        brightness: float = 0.0,
        contrast: float = 0.0,
    ):
        self.hflip = hflip
        self.vflip = vflip
        self.rotate90 = rotate90
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.hflip and np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()
            mask = mask[:, ::-1].copy()
        if self.vflip and np.random.rand() < 0.5:
            img = img[::-1, :, :].copy()
            mask = mask[::-1, :].copy()
        if self.rotate90:
            k = int(np.random.randint(0, 4))
            if k:
                img = np.rot90(img, k, axes=(0, 1)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()
        if self.brightness > 0:
            img = np.clip(img + np.random.uniform(-self.brightness, self.brightness), 0.0, 1.0)
        if self.contrast > 0:
            factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            img = np.clip((img - 0.5) * factor + 0.5, 0.0, 1.0)
        return img, mask
