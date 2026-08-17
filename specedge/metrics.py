"""
边缘质量评估指标 (从光刻 LER 推广):
  - ERS: Edge Roughness Score (3sigma, 越小越好)
  - SEQ: Spectral Edge Quality (低频/高频能量比, 越大越好)
  - BF1: Boundary F1
"""

import numpy as np
from scipy import ndimage


def extract_boundary(mask, width=1):
    """从二值 mask 提取边界像素。"""
    binary = mask > 0.5
    eroded = ndimage.binary_erosion(binary, iterations=width)
    return (binary & ~eroded).astype(np.float32)


def edge_roughness_score(pred_mask, gt_mask):
    """ERS: 借鉴光刻 LER 的 3sigma 定义。"""
    pred_bd = extract_boundary(pred_mask)
    gt_bd = extract_boundary(gt_mask)
    if pred_bd.sum() == 0 or gt_bd.sum() == 0:
        return float("inf")
    gt_dist = ndimage.distance_transform_edt(1 - gt_bd)
    distances = gt_dist[pred_bd > 0]
    return float(3.0 * distances.std())


def spectral_edge_quality(pred_mask, cutoff_ratio=0.25):
    """SEQ: PSD 低频/高频能量比。"""
    from scipy.ndimage import sobel
    edge_x = sobel(pred_mask.astype(np.float64), axis=1)
    edge_y = sobel(pred_mask.astype(np.float64), axis=0)
    edge_map = np.sqrt(edge_x ** 2 + edge_y ** 2)
    if edge_map.max() < 1e-8:
        return 0.0

    fft_2d = np.fft.fftshift(np.fft.fft2(edge_map))
    psd = np.abs(fft_2d) ** 2
    H, W = psd.shape
    Y, X = np.ogrid[:H, :W]
    radius = np.sqrt((X - W / 2.0) ** 2 + (Y - H / 2.0) ** 2)
    max_r = radius.max()

    low_energy = psd[radius <= cutoff_ratio * max_r].sum()
    high_energy = psd[radius > cutoff_ratio * max_r].sum()
    return float(low_energy / (high_energy + 1e-8))


def boundary_f1(pred_mask, gt_mask, tolerance=2):
    """BF1: 在给定容差内的边缘 F1。"""
    pred_bd = extract_boundary(pred_mask)
    gt_bd = extract_boundary(gt_mask)
    if pred_bd.sum() == 0 and gt_bd.sum() == 0:
        return 1.0
    if pred_bd.sum() == 0 or gt_bd.sum() == 0:
        return 0.0
    gt_dist = ndimage.distance_transform_edt(1 - gt_bd)
    pred_dist = ndimage.distance_transform_edt(1 - pred_bd)
    precision = (gt_dist[pred_bd > 0] <= tolerance).mean()
    recall = (pred_dist[gt_bd > 0] <= tolerance).mean()
    if precision + recall < 1e-8:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_iou(pred_mask, gt_mask):
    p, g = pred_mask > 0.5, gt_mask > 0.5
    inter = (p & g).sum()
    union = (p | g).sum()
    return 1.0 if union == 0 else float(inter / union)


def evaluate_all(pred_mask, gt_mask):
    """一次性计算所有指标。"""
    return {
        "iou": compute_iou(pred_mask, gt_mask),
        "bf1": boundary_f1(pred_mask, gt_mask),
        "ers": edge_roughness_score(pred_mask, gt_mask),
        "seq": spectral_edge_quality(pred_mask),
    }
