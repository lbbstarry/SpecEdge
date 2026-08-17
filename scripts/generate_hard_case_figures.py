"""Generate compact hard-set case figures for paper/debugging."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="dataset/litho_hard")
    p.add_argument("--pred-root", default="output/hard_eval")
    p.add_argument("--output-dir", default="output/hard_eval/figures")
    p.add_argument("--split", default="hard")
    p.add_argument("--cases", nargs="+", default=["9", "10", "14", "16", "22", "7"])
    p.add_argument("--models", nargs="+", default=["unet", "hrnet", "segformer"])
    p.add_argument("--thumb-size", type=int, default=180)
    return p.parse_args()


def load_panel(path: Path, label: str, thumb: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((thumb, thumb), Image.BILINEAR)
    panel = Image.new("RGB", (thumb, thumb + 26), "white")
    panel.paste(image, ((thumb - image.width) // 2, 0))
    draw = ImageDraw.Draw(panel)
    draw.text((4, thumb + 6), label, fill=(0, 0, 0))
    return panel


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    pred_root = Path(args.pred_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    columns: list[tuple[str, str]] = [
        ("SEM", str(data_root / "images" / args.split / "{case}.png")),
        ("GT mask", str(data_root / "masks" / args.split / "{case}.png")),
        ("manual layout", str(data_root / "layout_manual" / args.split / "{case}.png")),
    ]
    for model in args.models:
        columns.append((model, str(pred_root / model / "preds" / "masks" / "{case}.png")))

    thumb = args.thumb_size
    rows: list[list[Image.Image]] = []
    for case in args.cases:
        row: list[Image.Image] = []
        for title, pattern in columns:
            path = Path(pattern.format(case=case))
            row.append(load_panel(path, f"{case} {title}", thumb))
        rows.append(row)

    width = thumb * len(columns)
    height = (thumb + 26) * len(rows)
    sheet = Image.new("RGB", (width, height), "white")
    for r, row in enumerate(rows):
        for c, panel in enumerate(row):
            sheet.paste(panel, (c * thumb, r * (thumb + 26)))

    out = out_dir / "hard_failure_contact.png"
    sheet.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
