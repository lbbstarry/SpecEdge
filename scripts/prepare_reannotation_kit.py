"""Build the blinded re-annotation kit (revision plan v4, E2).

Takes the 5 SegFormer worst-case Extreme samples plus 3 non-failure controls,
shuffles them under anonymous kit ids (K01..K08), and renders for each:

  A) the raw SEM image;
  B) the SEM image with SegFormer "extra foreground" connected components
     (pred AND NOT reference, area >= 64 px) outlined and numbered.

The annotator sees only A/B (never the LithoSeg reference mask) and judges each
numbered component as hallucination / missed-by-reference / ambiguous in
recording_sheet.csv. The kit-id -> sample mapping lives in _private/ and must
not be opened before annotation is finished.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
IMG_DIR = REPO / "dataset/litho_hard/images/hard"
REF_DIR = REPO / "dataset/litho_hard/masks/hard"
PRED_DIR = REPO / "output/hard_eval/segformer/preds/masks"
OUT = REPO / "output/revision_v4/reannotation_kit"
PRIVATE = OUT / "_private"

WORST5 = ["9", "10", "14", "16", "22"]
CONTROLS = ["0", "13", "27"]
MIN_AREA = 64
SEED = 7


def load_bin(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 127).astype(np.uint8)


def resize_like(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    h, w = shape
    out = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST))
    return (out > 127).astype(np.uint8)


def render(name: str, kit_id: str) -> list[dict[str, object]]:
    sem = np.asarray(Image.open(IMG_DIR / f"{name}.png").convert("L"))
    ref = load_bin(REF_DIR / f"{name}.png")
    pred = resize_like(load_bin(PRED_DIR / f"{name}.png"), ref.shape)
    extra = ((pred == 1) & (ref == 0)).astype(np.uint8)
    ncc, labels, stats, cents = cv2.connectedComponentsWithStats(extra, 8)
    Image.fromarray(sem).save(OUT / f"{kit_id}_A_sem.png")
    vis = cv2.cvtColor(sem, cv2.COLOR_GRAY2BGR)
    rows = []
    comp_idx = 0
    for j in range(1, ncc):
        x0, y0, w, h, area = stats[j]
        if area < MIN_AREA:
            continue
        comp_idx += 1
        comp = (labels == j).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
        cx, cy = int(cents[j][0]), int(cents[j][1])
        cv2.putText(vis, str(comp_idx), (min(cx, vis.shape[1] - 40), max(cy, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        rows.append({"kit_id": kit_id, "component_id": comp_idx, "area_px": int(area),
                     "verdict": "", "notes": ""})
    cv2.imwrite(str(OUT / f"{kit_id}_B_components.png"), vis)
    if not rows:
        rows.append({"kit_id": kit_id, "component_id": 0, "area_px": 0,
                     "verdict": "no_component_flagged", "notes": ""})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(exist_ok=True)
    samples = [(n, False) for n in WORST5] + [(n, True) for n in CONTROLS]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(samples))
    mapping = {}
    sheet_rows: list[dict[str, object]] = []
    for rank, idx in enumerate(order, start=1):
        name, is_control = samples[idx]
        kit_id = f"K{rank:02d}"
        mapping[kit_id] = {"name": name, "is_control": is_control}
        sheet_rows.extend(render(name, kit_id))
    with open(OUT / "recording_sheet.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kit_id", "component_id", "area_px", "verdict", "notes"])
        writer.writeheader()
        writer.writerows(sheet_rows)
    json.dump(mapping, open(PRIVATE / "mapping.json", "w"), indent=2)
    readme = """# 盲法 re-annotation 标注包（修改计划 v4 / E2）

## 标注协议

1. 只看 `K0X_A_sem.png`（原始 SEM）与 `K0X_B_components.png`（红色轮廓 = 待判定区域，橙色数字 = 区域编号）。
2. **不要打开 `_private/`**（其中是样本身份映射，标注完成后才能看）。
3. 对每个编号区域，在 `recording_sheet.csv` 的 `verdict` 列填一项：
   - `hallucination` — SEM 上该区域没有合理的 photoresist 证据，预测是错的；
   - `missed_by_reference` — SEM 上该区域有合理的 photoresist 证据，参考似乎漏标；
   - `ambiguous` — 无法判断。
4. 判断依据只有 SEM 图像本身（亮度、纹理、与邻近线条的连续性），不参考任何 mask。
5. 8 张图中混有非失效对照样本，请按同一标准判定，不要猜测哪张是对照。
6. 全部填完后再打开 `_private/mapping.json` 解盲并汇总到论文表 6。

## 建议

- 间隔 24 小时做第二遍自检（或请第二人标注一遍），记录不一致项。
- 标注时间预计 20-30 分钟。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    n_comp = sum(1 for r in sheet_rows if r["component_id"] != 0)
    print(f"kit ready: {len(mapping)} samples, {n_comp} components to judge -> {OUT}")


if __name__ == "__main__":
    main()
