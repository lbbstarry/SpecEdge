"""分割边缘评估指标合集.

包含:
  - iou / dice
  - boundary_f1: 容差像素内的边界 F1
  - hausdorff95: 边界点 95 分位 Hausdorff 距离
  - edge_psd_hf_ratio: 边缘 2D PSD 高频能量占比 (本工作核心区分项)

约定: pred / gt 为 numpy 二值 mask, shape (H, W), 取值 {0, 1}.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


# ---------- 基础指标 ----------


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    inter = float(((pred == 1) & (gt == 1)).sum())
    union = float(((pred == 1) | (gt == 1)).sum())
    return inter / (union + eps)


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    inter = float(((pred == 1) & (gt == 1)).sum())
    return 2.0 * inter / (pred.sum() + gt.sum() + eps)


# ---------- 边界相关 ----------


def _boundary(mask: np.ndarray) -> np.ndarray:
    eroded = ndi.binary_erosion(mask.astype(bool))
    return np.logical_and(mask.astype(bool), np.logical_not(eroded))


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance_px: int = 2, eps: float = 1e-7) -> float:
    """容差 tolerance_px 的边界 F1 (BF score)."""
    pb = _boundary(pred)
    gb = _boundary(gt)
    if not pb.any() and not gb.any():
        return 1.0
    if not pb.any() or not gb.any():
        return 0.0

    pb_dilated = ndi.binary_dilation(pb, iterations=tolerance_px)
    gb_dilated = ndi.binary_dilation(gb, iterations=tolerance_px)

    precision = float(np.logical_and(pb, gb_dilated).sum()) / (float(pb.sum()) + eps)
    recall = float(np.logical_and(gb, pb_dilated).sum()) / (float(gb.sum()) + eps)
    if precision + recall < eps:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hausdorff95(pred: np.ndarray, gt: np.ndarray) -> float:
    pb = _boundary(pred)
    gb = _boundary(gt)
    if not pb.any() or not gb.any():
        return float("inf")

    # 距离变换给出每个像素到最近边界点的距离, 在对侧边界点上采样
    dt_to_gt = ndi.distance_transform_edt(np.logical_not(gb))
    dt_to_pred = ndi.distance_transform_edt(np.logical_not(pb))

    d_pred_to_gt = dt_to_gt[pb]
    d_gt_to_pred = dt_to_pred[gb]
    return float(max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95)))


# ---------- PSD 高频能量比 ----------


def edge_map(mask: np.ndarray) -> np.ndarray:
    """Sobel 边缘强度图."""
    gx = ndi.sobel(mask.astype(np.float32), axis=1)
    gy = ndi.sobel(mask.astype(np.float32), axis=0)
    return np.hypot(gx, gy)


def _radial_psd(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (径向频率归一到 [0,1], 径向平均功率谱)."""
    f = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(f) ** 2

    h, w = img.shape
    cy, cx = h / 2, w / 2
    y, x = np.indices(img.shape)
    r = np.hypot(y - cy, x - cx)
    r_int = r.astype(np.int32)

    tbin = np.bincount(r_int.ravel(), power.ravel())
    nr = np.bincount(r_int.ravel())
    radial = tbin / np.maximum(nr, 1)

    r_max = min(cy, cx)
    radial = radial[: int(r_max)]
    freq = np.linspace(0, 1, num=radial.shape[0], endpoint=False)
    return freq, radial


def edge_psd_hf_ratio(mask: np.ndarray, hf_cutoff_ratio: float = 0.5) -> float:
    """边缘 PSD 在 [hf_cutoff_ratio, 1.0] 频段的能量占比.

    值越大代表边缘越锯齿. baseline 通常显著高于 GT.
    """
    e = edge_map(mask)
    if e.sum() < 1e-6:
        return 0.0
    freq, psd = _radial_psd(e)
    total = float(psd.sum())
    if total < 1e-12:
        return 0.0
    hf_mask = freq >= hf_cutoff_ratio
    return float(psd[hf_mask].sum()) / total


# ---------- 聚合 ----------


def evaluate_pair(
    pred: np.ndarray,
    gt: np.ndarray,
    boundary_tolerance_px: int = 2,
    hf_cutoff_ratio: float = 0.5,
) -> dict[str, float]:
    return {
        "iou": iou_score(pred, gt),
        "dice": dice_score(pred, gt),
        "boundary_f1": boundary_f1(pred, gt, tolerance_px=boundary_tolerance_px),
        "hausdorff95": hausdorff95(pred, gt),
        "edge_psd_hf_ratio_pred": edge_psd_hf_ratio(pred, hf_cutoff_ratio),
        "edge_psd_hf_ratio_gt": edge_psd_hf_ratio(gt, hf_cutoff_ratio),
    }
