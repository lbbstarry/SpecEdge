"""Evaluate mask-derived lithography metrology indicators.

Examples:
    # GT-only diagnostics
    python scripts/eval_metrology.py \
        --gt-dir dataset/litho/masks/test \
        --output output/metrology/gt_test_metrics.csv

    # Pred-vs-GT metrology error
    python scripts/eval_metrology.py \
        --gt-dir dataset/litho/masks/test \
        --pred-dir output/baselines/segformer/preds/masks \
        --output output/metrology/segformer_test_metrics.csv \
        --summary output/metrology/segformer_test_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from specedge.metrology import evaluate_metrology_pair, mask_metrology, summarize_pair_rows


IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", required=True, help="Directory containing GT masks")
    p.add_argument("--pred-dir", default=None, help="Optional directory containing predicted masks")
    p.add_argument("--output", required=True, help="CSV output path")
    p.add_argument("--summary", default=None, help="Optional JSON summary output path")
    p.add_argument("--min-area", type=int, default=64, help="Minimum connected-component area to measure")
    return p.parse_args()


def mask_files(mask_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(mask_dir.iterdir()):
        if path.suffix.lower() in IMG_EXTS:
            files[path.stem] = path
    return files


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def resize_like(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    h, w = shape
    return np.asarray(Image.fromarray(mask).resize((w, h), Image.NEAREST))


def stringify(value: object) -> object:
    if isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(float(value)) else f"{float(value):.8g}"
    return value


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("no rows to write")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: stringify(row.get(k, "")) for k in fieldnames})


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    output = Path(args.output)

    gt_files = mask_files(gt_dir)
    if not gt_files:
        raise RuntimeError(f"no mask files found in {gt_dir}")

    rows: list[dict[str, object]] = []
    if pred_dir is None:
        for name, gt_path in gt_files.items():
            row = {"name": name, **mask_metrology(load_mask(gt_path), min_area=args.min_area)}
            rows.append(row)
        summary = None
    else:
        pred_files = mask_files(pred_dir)
        for name, gt_path in gt_files.items():
            pred_path = pred_files.get(name)
            if pred_path is None:
                continue
            pred_mask = load_mask(pred_path)
            gt_mask = resize_like(load_mask(gt_path), pred_mask.shape)
            row = {
                "name": name,
                **evaluate_metrology_pair(
                    pred_mask,
                    gt_mask,
                    min_area=args.min_area,
                ),
            }
            rows.append(row)
        if not rows:
            raise RuntimeError(f"no paired pred/gt masks found in {pred_dir} and {gt_dir}")
        summary = summarize_pair_rows(rows)

    write_csv(rows, output)

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": "pair" if pred_dir else "gt_only",
            "num_rows": len(rows),
            "gt_dir": str(gt_dir),
            "pred_dir": str(pred_dir) if pred_dir else None,
            "summary": summary,
        }
        with summary_path.open("w") as f:
            json.dump(payload, f, indent=2)

    print(f"wrote {len(rows)} rows -> {output}")
    if args.summary:
        print(f"summary -> {args.summary}")


if __name__ == "__main__":
    main()
