"""Mask-to-metrology utilities for lithography SEM segmentation.

The functions in this module treat a binary mask as the measurement contour
source and extract first-pass process indicators: CD, LWR, LER, 1D edge PSD,
and simple topology risk signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi


EPS = 1e-8


@dataclass
class ComponentProfile:
    component_id: int
    length: int
    cd_mean: float
    cd_std: float
    lwr_3sigma: float
    ler_low_3sigma: float
    ler_high_3sigma: float
    ler_mean_3sigma: float
    edge_psd_hf_ratio: float
    necking_score: float
    bulging_score: float


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert an arbitrary mask array to {0, 1} uint8."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    threshold = 127 if arr.max(initial=0) > 1 else 0.5
    return (arr > threshold).astype(np.uint8)


def connected_components(mask: np.ndarray, min_area: int = 64) -> tuple[np.ndarray, list[int]]:
    binary = binarize_mask(mask).astype(bool)
    labels, num = ndi.label(binary)
    keep: list[int] = []
    for label in range(1, num + 1):
        area = int((labels == label).sum())
        if area >= min_area:
            keep.append(label)
    return labels, keep


def _component_bbox(component: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(component)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def estimate_dominant_orientation(mask: np.ndarray, min_area: int = 64) -> str:
    """Return horizontal, vertical, empty, or complex based on component shapes."""
    labels, keep = connected_components(mask, min_area=min_area)
    if not keep:
        return "empty"

    horizontal_area = 0.0
    vertical_area = 0.0
    ambiguous_area = 0.0
    for label in keep:
        component = labels == label
        y0, x0, y1, x1 = _component_bbox(component)
        h = max(1, y1 - y0)
        w = max(1, x1 - x0)
        area = float(component.sum())
        aspect = w / h
        if aspect >= 1.4:
            horizontal_area += area
        elif aspect <= 1 / 1.4:
            vertical_area += area
        else:
            ambiguous_area += area

    total = horizontal_area + vertical_area + ambiguous_area
    if horizontal_area / max(total, EPS) > 0.6:
        return "horizontal"
    if vertical_area / max(total, EPS) > 0.6:
        return "vertical"
    return "complex"


def _detrend(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return values - values.mean() if values.size else values
    x = np.arange(values.size, dtype=np.float64)
    coeff = np.polyfit(x, values, deg=1)
    trend = coeff[0] * x + coeff[1]
    return values - trend


def _psd_hf_ratio_1d(values: np.ndarray, hf_cutoff_ratio: float = 0.5) -> float:
    residual = _detrend(values)
    if residual.size < 4 or float(np.abs(residual).sum()) < EPS:
        return 0.0
    fft = np.fft.rfft(residual)
    power = np.abs(fft) ** 2
    if power.size <= 1:
        return 0.0
    power = power[1:]  # drop DC
    total = float(power.sum())
    if total < EPS:
        return 0.0
    freq = np.linspace(0, 1, num=power.size, endpoint=True)
    return float(power[freq >= hf_cutoff_ratio].sum() / total)


def _safe_float(value: float | np.floating) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def _profile_from_component(component: np.ndarray, orientation: str, component_id: int) -> ComponentProfile | None:
    """Extract width and edge profiles for one connected component."""
    y0, x0, y1, x1 = _component_bbox(component)
    crop = component[y0:y1, x0:x1]

    widths: list[float] = []
    low_edges: list[float] = []
    high_edges: list[float] = []

    if orientation == "horizontal":
        for xi in range(crop.shape[1]):
            ys = np.nonzero(crop[:, xi])[0]
            if ys.size == 0:
                continue
            low = float(ys.min() + y0)
            high = float(ys.max() + y0)
            low_edges.append(low)
            high_edges.append(high)
            widths.append(high - low + 1.0)
    elif orientation == "vertical":
        for yi in range(crop.shape[0]):
            xs = np.nonzero(crop[yi, :])[0]
            if xs.size == 0:
                continue
            low = float(xs.min() + x0)
            high = float(xs.max() + x0)
            low_edges.append(low)
            high_edges.append(high)
            widths.append(high - low + 1.0)
    else:
        return None

    if len(widths) < 8:
        return None

    w = np.asarray(widths, dtype=np.float64)
    low_residual = _detrend(np.asarray(low_edges, dtype=np.float64))
    high_residual = _detrend(np.asarray(high_edges, dtype=np.float64))
    low_ler = 3.0 * float(low_residual.std())
    high_ler = 3.0 * float(high_residual.std())
    mean_width = float(w.mean())
    min_width = float(w.min())
    max_width = float(w.max())
    width_std = float(w.std())
    return ComponentProfile(
        component_id=component_id,
        length=len(widths),
        cd_mean=_safe_float(mean_width),
        cd_std=_safe_float(width_std),
        lwr_3sigma=_safe_float(3.0 * width_std),
        ler_low_3sigma=_safe_float(low_ler),
        ler_high_3sigma=_safe_float(high_ler),
        ler_mean_3sigma=_safe_float((low_ler + high_ler) / 2.0),
        edge_psd_hf_ratio=_safe_float(
            (_psd_hf_ratio_1d(low_residual) + _psd_hf_ratio_1d(high_residual)) / 2.0
        ),
        necking_score=_safe_float((mean_width - min_width) / max(mean_width, EPS)),
        bulging_score=_safe_float((max_width - mean_width) / max(mean_width, EPS)),
    )


def extract_component_profiles(
    mask: np.ndarray,
    orientation: str | None = None,
    min_area: int = 64,
) -> list[ComponentProfile]:
    binary = binarize_mask(mask)
    orientation = orientation or estimate_dominant_orientation(binary, min_area=min_area)
    if orientation not in {"horizontal", "vertical"}:
        return []

    labels, keep = connected_components(binary, min_area=min_area)
    profiles: list[ComponentProfile] = []
    for label in keep:
        profile = _profile_from_component(labels == label, orientation, component_id=label)
        if profile is not None:
            profiles.append(profile)
    return profiles


def _weighted_mean(profiles: Iterable[ComponentProfile], attr: str) -> float:
    profiles = list(profiles)
    values = np.asarray([getattr(p, attr) for p in profiles], dtype=np.float64)
    weights = np.asarray([p.length for p in profiles], dtype=np.float64)
    valid = np.isfinite(values) & (weights > 0)
    if not valid.any():
        return float("nan")
    return _safe_float(np.average(values[valid], weights=weights[valid]))


def _max_value(profiles: Iterable[ComponentProfile], attr: str) -> float:
    values = np.asarray([getattr(p, attr) for p in profiles], dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return float("nan")
    return _safe_float(values[valid].max())


def mask_metrology(mask: np.ndarray, min_area: int = 64) -> dict[str, float | str | int]:
    """Compute first-pass metrology indicators from a binary mask."""
    binary = binarize_mask(mask)
    labels, keep = connected_components(binary, min_area=min_area)
    orientation = estimate_dominant_orientation(binary, min_area=min_area)
    profiles = extract_component_profiles(binary, orientation=orientation, min_area=min_area)
    fg_ratio = float(binary.mean())

    result: dict[str, float | str | int] = {
        "status": "ok" if profiles else ("empty" if orientation == "empty" else "complex"),
        "orientation": orientation,
        "foreground_ratio": _safe_float(fg_ratio),
        "component_count": len(keep),
        "profile_count": len(profiles),
        "profile_total_length": int(sum(p.length for p in profiles)),
        "cd_mean": _weighted_mean(profiles, "cd_mean"),
        "cd_std": _weighted_mean(profiles, "cd_std"),
        "lwr_3sigma": _weighted_mean(profiles, "lwr_3sigma"),
        "ler_low_3sigma": _weighted_mean(profiles, "ler_low_3sigma"),
        "ler_high_3sigma": _weighted_mean(profiles, "ler_high_3sigma"),
        "ler_mean_3sigma": _weighted_mean(profiles, "ler_mean_3sigma"),
        "edge_psd_hf_ratio_1d": _weighted_mean(profiles, "edge_psd_hf_ratio"),
        "necking_score": _max_value(profiles, "necking_score"),
        "bulging_score": _max_value(profiles, "bulging_score"),
    }
    return result


def _metric_error(pred_value: object, gt_value: object) -> float:
    try:
        pred_f = float(pred_value)
        gt_f = float(gt_value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(pred_f) or not np.isfinite(gt_f):
        return float("nan")
    return _safe_float(pred_f - gt_f)


def evaluate_metrology_pair(pred_mask: np.ndarray, gt_mask: np.ndarray, min_area: int = 64) -> dict[str, object]:
    """Compare pred-derived metrology against GT-derived reference metrology."""
    pred = mask_metrology(pred_mask, min_area=min_area)
    gt = mask_metrology(gt_mask, min_area=min_area)

    row: dict[str, object] = {}
    for key, value in pred.items():
        row[f"pred_{key}"] = value
    for key, value in gt.items():
        row[f"gt_{key}"] = value

    metric_keys = [
        "foreground_ratio",
        "component_count",
        "profile_count",
        "cd_mean",
        "cd_std",
        "lwr_3sigma",
        "ler_low_3sigma",
        "ler_high_3sigma",
        "ler_mean_3sigma",
        "edge_psd_hf_ratio_1d",
        "necking_score",
        "bulging_score",
    ]
    for key in metric_keys:
        signed = _metric_error(pred.get(key), gt.get(key))
        row[f"err_{key}"] = signed
        row[f"abs_err_{key}"] = abs(signed) if np.isfinite(signed) else float("nan")

    pred_cc = int(pred.get("component_count", 0))
    gt_cc = int(gt.get("component_count", 0))
    row["component_count_error"] = pred_cc - gt_cc
    row["bridge_candidate"] = bool(pred_cc < gt_cc)
    row["open_candidate"] = bool(pred_cc > gt_cc)

    gt_cd = float(gt.get("cd_mean", float("nan")))
    pred_cd = float(pred.get("cd_mean", float("nan")))
    if np.isfinite(gt_cd) and np.isfinite(pred_cd) and gt_cd > EPS:
        row["rel_err_cd_mean"] = _safe_float((pred_cd - gt_cd) / gt_cd)
    else:
        row["rel_err_cd_mean"] = float("nan")

    neck = float(pred.get("necking_score", float("nan")))
    bulge = float(pred.get("bulging_score", float("nan")))
    row["necking_candidate"] = bool(np.isfinite(neck) and neck > 0.25)
    row["bulging_candidate"] = bool(np.isfinite(bulge) and bulge > 0.25)
    return row


def summarize_pair_rows(rows: list[dict[str, object]]) -> dict[str, float | int]:
    """Aggregate metrology pair rows into paper-friendly summary statistics."""
    summary: dict[str, float | int] = {"num_samples": len(rows)}
    metrics = [
        "cd_mean",
        "cd_std",
        "lwr_3sigma",
        "ler_mean_3sigma",
        "edge_psd_hf_ratio_1d",
        "foreground_ratio",
        "component_count",
    ]
    for metric in metrics:
        abs_key = f"abs_err_{metric}"
        values = np.asarray([float(r.get(abs_key, float("nan"))) for r in rows], dtype=np.float64)
        valid = values[np.isfinite(values)]
        summary[f"{metric}_mae"] = _safe_float(valid.mean()) if valid.size else float("nan")
        summary[f"{metric}_rmse"] = _safe_float(sqrt(float((valid ** 2).mean()))) if valid.size else float("nan")

        pred_values = np.asarray([float(r.get(f"pred_{metric}", float("nan"))) for r in rows], dtype=np.float64)
        gt_values = np.asarray([float(r.get(f"gt_{metric}", float("nan"))) for r in rows], dtype=np.float64)
        ok = np.isfinite(pred_values) & np.isfinite(gt_values)
        if ok.sum() >= 2 and pred_values[ok].std() > EPS and gt_values[ok].std() > EPS:
            summary[f"{metric}_pearson"] = _safe_float(np.corrcoef(pred_values[ok], gt_values[ok])[0, 1])
        else:
            summary[f"{metric}_pearson"] = float("nan")

    summary["bridge_candidate_count"] = int(sum(bool(r.get("bridge_candidate")) for r in rows))
    summary["open_candidate_count"] = int(sum(bool(r.get("open_candidate")) for r in rows))
    summary["necking_candidate_count"] = int(sum(bool(r.get("necking_candidate")) for r in rows))
    summary["bulging_candidate_count"] = int(sum(bool(r.get("bulging_candidate")) for r in rows))
    return summary

