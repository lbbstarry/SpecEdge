"""E11 -- Classical contour-extraction baselines for the mask-to-metrology pipeline.

Two classical, learning-free segmentation frontends are compared against the four
neural frontends (U-Net, DeepLabV3+, HRNet, SegFormer) on both the in-distribution
test split (n=60) and the Extreme process-window split (n=65).

Baseline A -- Otsu global thresholding + morphological cleanup.
Baseline B -- Canny edge detection + morphological closing + flood-fill from border.

Both baselines emit a binary mask on the same H x W grid as the reference mask,
which is then routed through specedge.metrology.evaluate_metrology_pair on
identical footing with the neural frontends. IoU / BF1 are computed alongside
CD / LWR / LER MAE so the two baselines slot straight into Table II and the
Extreme summary table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from specedge.metrics import boundary_f1  # noqa: E402
from specedge.metrology import evaluate_metrology_pair  # noqa: E402


OUT_ROOT = REPO_ROOT / "output" / "revision_v4" / "e11_classical"


def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.uint8)


def resize_nn(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    h, w = shape
    return np.asarray(Image.fromarray(mask).resize((w, h), Image.NEAREST))


def bright_is_foreground(gray: np.ndarray, mask: np.ndarray) -> bool:
    if mask.sum() == 0 or (~mask).sum() == 0:
        return True
    fg_mean = float(gray[mask].mean())
    bg_mean = float(gray[~mask].mean())
    return fg_mean >= bg_mean


def otsu_mask(gray: np.ndarray) -> np.ndarray:
    smoothed = cv2.medianBlur(gray, 5)
    _, binary = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = binary > 0
    if not bright_is_foreground(gray, binary):
        binary = ~binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_u8 = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1)
    binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return (binary_u8 > 0).astype(np.uint8)


def canny_fill_mask(gray: np.ndarray) -> np.ndarray:
    smoothed = cv2.medianBlur(gray, 5)
    otsu_val, _ = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = int(max(1, otsu_val))
    low = int(max(1, 0.5 * high))
    edges = cv2.Canny(smoothed, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    inv = (edges_closed == 0).astype(np.uint8)
    h, w = inv.shape
    flood = inv.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for x in (0, w - 1):
        for y in range(h):
            if flood[y, x] == 1:
                cv2.floodFill(flood, ff_mask, (x, y), 2)
    for y in (0, h - 1):
        for x in range(w):
            if flood[y, x] == 1:
                cv2.floodFill(flood, ff_mask, (x, y), 2)

    interior = (flood == 1).astype(np.uint8)
    binary = interior > 0
    if not bright_is_foreground(gray, binary):
        binary = (~binary) & (edges_closed == 0)
    binary_u8 = binary.astype(np.uint8)
    binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    return (binary_u8 > 0).astype(np.uint8)


def iou_bf1(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    p = pred > 0
    g = gt > 0
    inter = float(np.logical_and(p, g).sum())
    union = float(np.logical_or(p, g).sum())
    iou = inter / union if union > 0 else float("nan")
    bf1 = float(boundary_f1(p.astype(np.uint8), g.astype(np.uint8), tolerance=2))
    return iou, bf1


def mae(values: list[float]) -> float:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.abs(arr).mean())


def _safe_float(v) -> float:
    try:
        val = float(v)
        if np.isfinite(val):
            return val
    except (TypeError, ValueError):
        pass
    return float("nan")


def run_split(name: str, image_dir: Path, mask_dir: Path, out_root: Path) -> dict[str, dict[str, float]]:
    image_files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() == ".png")
    mask_files = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() == ".png"}

    per_method: dict[str, dict[str, list]] = {
        "otsu": {"iou": [], "bf1": [], "cd_err": [], "lwr_err": [], "ler_err": [],
                  "cd_ok": [], "profile_ok": []},
        "canny": {"iou": [], "bf1": [], "cd_err": [], "lwr_err": [], "ler_err": [],
                  "cd_ok": [], "profile_ok": []},
    }

    for img_path in image_files:
        stem = img_path.stem
        if stem not in mask_files:
            continue
        gray = load_gray(img_path)
        gt = load_gray(mask_files[stem])
        gt_bin = (gt > 127).astype(np.uint8)
        gt_bin = resize_nn(gt_bin, gray.shape)

        for method, fn in (("otsu", otsu_mask), ("canny", canny_fill_mask)):
            pred = fn(gray)
            out_dir = out_root / method / "preds" / "masks" / name
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray((pred * 255).astype(np.uint8)).save(out_dir / f"{stem}.png")

            iou, bf1 = iou_bf1(pred, gt_bin)
            per_method[method]["iou"].append(iou)
            per_method[method]["bf1"].append(bf1)

            metro = evaluate_metrology_pair(
                pred.astype(np.uint8) * 255,
                gt_bin.astype(np.uint8) * 255,
                min_area=64,
            )
            cd_err = _safe_float(metro.get("abs_err_cd_mean"))
            lwr_err = _safe_float(metro.get("abs_err_lwr_3sigma"))
            ler_err = _safe_float(metro.get("abs_err_ler_mean_3sigma"))
            per_method[method]["cd_err"].append(cd_err)
            per_method[method]["lwr_err"].append(lwr_err)
            per_method[method]["ler_err"].append(ler_err)
            per_method[method]["cd_ok"].append(1.0 if np.isfinite(cd_err) else 0.0)
            profile_ok = np.isfinite(lwr_err) and np.isfinite(ler_err)
            per_method[method]["profile_ok"].append(1.0 if profile_ok else 0.0)

    summary: dict[str, dict[str, float]] = {}
    for method, buckets in per_method.items():
        summary[method] = {
            "n": len(buckets["iou"]),
            "iou_mean": mae(buckets["iou"]),
            "bf1_mean": mae(buckets["bf1"]),
            "cd_mae_px": mae(buckets["cd_err"]),
            "lwr_mae_px": mae(buckets["lwr_err"]),
            "ler_mae_px": mae(buckets["ler_err"]),
            "cd_ok_rate": float(np.mean(buckets["cd_ok"])) if buckets["cd_ok"] else float("nan"),
            "profile_ok_rate": float(np.mean(buckets["profile_ok"])) if buckets["profile_ok"] else float("nan"),
        }
    return summary


def main() -> None:
    splits = [
        ("standard_test",
         REPO_ROOT / "dataset" / "litho" / "images" / "test",
         REPO_ROOT / "dataset" / "litho" / "masks" / "test"),
        ("extreme_hard",
         REPO_ROOT / "dataset" / "litho_hard" / "images" / "hard",
         REPO_ROOT / "dataset" / "litho_hard" / "masks" / "hard"),
    ]
    all_results: dict[str, dict] = {}
    for split_name, img_dir, mask_dir in splits:
        print(f"[e11] running {split_name}: images={img_dir}, masks={mask_dir}")
        result = run_split(split_name, img_dir, mask_dir, OUT_ROOT)
        all_results[split_name] = result
        for method, stats in result.items():
            print(
                f"  [{split_name}][{method}] "
                f"n={stats['n']} IoU={stats['iou_mean']:.3f} BF1={stats['bf1_mean']:.3f} "
                f"CD_MAE={stats['cd_mae_px']:.3f}px "
                f"LWR_MAE={stats['lwr_mae_px']:.3f}px "
                f"LER_MAE={stats['ler_mae_px']:.3f}px "
                f"cd_ok={stats['cd_ok_rate']:.2f} "
                f"profile_ok={stats['profile_ok_rate']:.2f}"
            )

    out_json = OUT_ROOT / "e11_classical_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[e11] wrote {out_json}")


if __name__ == "__main__":
    main()
