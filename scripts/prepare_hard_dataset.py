"""Prepare the manually curated hard lithography SEM set.

The raw hard set currently uses this layout:

    dataset/hard/
      ADI/*.jpg       SEM images
      mask/*.jpg      reference masks (JPEG-compressed binary masks)
      layout/*.jpg    manually drawn layout sketches
      seg_sem/*.jpg   SEM + contour overlay images for review

This script converts it into a normalized evaluation-only split:

    dataset/litho_hard/
      images/hard/*.png
      masks/hard/*.png
      layout_manual/hard/*.png
      overlays/hard/*.png
      metadata.csv

The manually drawn layout sketches are copied for qualitative review and
taxonomy annotation only; they are not treated as pixel-accurate design truth.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from specedge.metrology import mask_metrology


IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", default="dataset/hard")
    p.add_argument("--output-root", default="dataset/litho_hard")
    p.add_argument("--split", default="hard")
    p.add_argument("--threshold", type=int, default=127)
    p.add_argument("--layout-review-size", type=int, default=1024)
    return p.parse_args()


def files_by_stem(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    if not directory.is_dir():
        return files
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in IMG_EXTS:
            files[path.stem] = path
    return files


def save_rgb_png(src: Path, dst: Path) -> tuple[int, int]:
    image = Image.open(src).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst)
    return image.size


def save_binary_mask(src: Path, dst: Path, threshold: int) -> tuple[int, int, float]:
    mask = Image.open(src).convert("L")
    arr = np.asarray(mask)
    binary = (arr > threshold).astype(np.uint8) * 255
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary).save(dst)
    return mask.size[0], mask.size[1], float((binary > 0).mean())


def save_layout_review(src: Path, dst: Path, target_size: int) -> tuple[int, int]:
    layout = Image.open(src).convert("RGBA")
    original_size = layout.size
    # For review only: fit the manually drawn sketch into a square canvas so it
    # can be viewed alongside SEM/mask images. Do not use this for measurement.
    layout.thumbnail((target_size, target_size), Image.BILINEAR)
    canvas = Image.new("RGBA", (target_size, target_size), (255, 255, 255, 255))
    x = (target_size - layout.width) // 2
    y = (target_size - layout.height) // 2
    canvas.paste(layout, (x, y), layout)
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(dst)
    return original_size


def auto_review_reason(row: dict[str, object]) -> str:
    reasons: list[str] = []
    if row.get("status") != "ok":
        reasons.append(str(row.get("status")))
    try:
        if float(row.get("lwr_3sigma", 0.0)) >= 25.0:
            reasons.append("high_lwr")
        if float(row.get("ler_mean_3sigma", 0.0)) >= 14.0:
            reasons.append("high_ler")
        if float(row.get("edge_psd_hf_ratio_1d", 0.0)) >= 0.08:
            reasons.append("high_psd")
        if int(row.get("component_count", 0)) >= 12:
            reasons.append("many_components")
        if float(row.get("foreground_ratio", 0.0)) <= 0.15:
            reasons.append("sparse_pattern")
    except (TypeError, ValueError):
        pass
    return ";".join(reasons)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    split = args.split

    adi_files = files_by_stem(input_root / "ADI")
    mask_files = files_by_stem(input_root / "mask")
    layout_files = files_by_stem(input_root / "layout")
    overlay_files = files_by_stem(input_root / "seg_sem")
    stems = sorted(set(adi_files) & set(mask_files), key=lambda s: int(s) if s.isdigit() else s)

    if not stems:
        raise RuntimeError(f"no paired ADI/mask files under {input_root}")

    rows: list[dict[str, object]] = []
    for stem in stems:
        image_dst = output_root / "images" / split / f"{stem}.png"
        mask_dst = output_root / "masks" / split / f"{stem}.png"
        layout_dst = output_root / "layout_manual" / split / f"{stem}.png"
        overlay_dst = output_root / "overlays" / split / f"{stem}.png"

        image_w, image_h = save_rgb_png(adi_files[stem], image_dst)
        mask_w, mask_h, mask_fg_ratio = save_binary_mask(mask_files[stem], mask_dst, threshold=args.threshold)

        has_layout = stem in layout_files
        layout_w = layout_h = ""
        if has_layout:
            layout_w, layout_h = save_layout_review(layout_files[stem], layout_dst, args.layout_review_size)

        has_overlay = stem in overlay_files
        if has_overlay:
            save_rgb_png(overlay_files[stem], overlay_dst)

        metrics = mask_metrology(np.asarray(Image.open(mask_dst).convert("L")))
        row: dict[str, object] = {
            "id": stem,
            "image_path": str(image_dst),
            "mask_path": str(mask_dst),
            "layout_manual_path": str(layout_dst) if has_layout else "",
            "overlay_path": str(overlay_dst) if has_overlay else "",
            "image_width": image_w,
            "image_height": image_h,
            "mask_width": mask_w,
            "mask_height": mask_h,
            "layout_width_original": layout_w,
            "layout_height_original": layout_h,
            "has_manual_layout": has_layout,
            "has_overlay": has_overlay,
            "mask_foreground_ratio_binary": f"{mask_fg_ratio:.8g}",
            **metrics,
            "manual_pattern_type": "",
            "manual_risk_tags": "",
            "manual_notes": "",
        }
        row["auto_review_reason"] = auto_review_reason(row)
        rows.append(row)

    metadata_path = output_root / "metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with metadata_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} samples -> {output_root}")
    print(f"metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
