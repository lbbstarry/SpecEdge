"""E13: layout-referenced absolute DFM verdicts.

``specedge/metrology.py`` already emits ``bridge_candidate`` /
``open_candidate``, but defines them *relative to the reference mask*
(``pred_cc < gt_cc``), which makes them segmentation-error indicators rather
than design decisions. A fab has no reference mask at runtime; what it has is
the layout.

This script re-derives the topology verdict against **design intent** taken
from the manually drawn layout, so the decision chain becomes

    layout (design) -> SEM -> segmentation -> metrology -> DFM verdict

and each frontend can be scored on how often it would have produced the wrong
*design-side* call, with no reference mask as an input.

Verdicts (computed from one mask plus the layout only):

    OPEN     component count > N_design     a designed line broke apart
    BRIDGE   component count < N_design     designed lines merged
    NOMINAL  otherwise

The reference mask supplies ground truth for scoring and is never an input to
a verdict.

Layout polarity: the sketches are drawn dark-on-light, so strokes sit below
the Otsu threshold. Deciding polarity from the image mean (as the r_bbox
control in ``revision_v4_analysis.py`` originally did) inverts 47 of the 65
sketches, whose drawn area covers more than half the canvas.

Outputs under ``output/revision_v4/e13_layout_dfm/``:

    design_intent.csv   per-sample N_design and extraction diagnostics
    verdicts.csv        per-sample, per-frontend verdict
    summary.json        per-frontend confusion against layout-referenced truth

Usage::

    python scripts/e13_layout_dfm_verdict.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

LAYOUT_DIR = REPO_ROOT / "dataset/litho_hard/layout_manual/hard"
REF_DIR = REPO_ROOT / "dataset/litho_hard/masks/hard"
METADATA = REPO_ROOT / "dataset/litho_hard/metadata.csv"
PRED_DIR = REPO_ROOT / "output/hard_eval/{m}/preds/masks"
OUT_DIR = REPO_ROOT / "output/revision_v4/e13_layout_dfm"

MODELS = ("unet", "deeplabv3plus", "hrnet", "segformer")

# Components below this pixel area are drawing or segmentation specks rather
# than designed features. Chosen to match the scale of the min_area used by the
# paper's own spurious-component counter so the two analyses stay comparable.
MIN_COMPONENT_AREA = 200


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-area", type=int, default=MIN_COMPONENT_AREA)
    p.add_argument("--out", default=str(OUT_DIR))
    return p.parse_args()


def count_components(binary: np.ndarray, min_area: int) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    return sum(1 for j in range(1, n) if stats[j][4] >= min_area)


def load_mask(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if img is None else img > 127


def load_design(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img < threshold


def verdict(component_count: int, n_design: int) -> str:
    if component_count > n_design:
        return "OPEN"
    if component_count < n_design:
        return "BRIDGE"
    return "NOMINAL"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(METADATA).set_index("id")

    design_rows, verdict_rows = [], []
    for sample_id in sorted(meta.index):
        design = load_design(LAYOUT_DIR / f"{sample_id}.png")
        reference = load_mask(REF_DIR / f"{sample_id}.png")
        if design is None or reference is None:
            continue

        n_design = count_components(design, args.min_area)
        n_reference = count_components(reference, args.min_area)

        design_rows.append(
            {
                "id": sample_id,
                "n_design": n_design,
                "n_reference": n_reference,
                "design_fg": float(design.mean()),
                "reference_fg": float(reference.mean()),
                "metadata_component_count": int(meta.loc[sample_id, "component_count"]),
                # A design/silicon disagreement is a genuine process defect, not
                # an extraction failure; kept so it can be reported separately.
                "design_silicon_mismatch": n_design != n_reference,
            }
        )

        row = {
            "id": sample_id,
            "n_design": n_design,
            "n_reference": n_reference,
            "truth": verdict(n_reference, n_design),
        }
        for model in MODELS:
            pred = load_mask(Path(str(PRED_DIR).format(m=model)) / f"{sample_id}.png")
            if pred is None:
                row[f"{model}_n"], row[f"{model}_verdict"] = np.nan, None
                continue
            n_pred = count_components(pred, args.min_area)
            row[f"{model}_n"] = n_pred
            row[f"{model}_verdict"] = verdict(n_pred, n_design)
        verdict_rows.append(row)

    design = pd.DataFrame(design_rows)
    verdicts = pd.DataFrame(verdict_rows)
    design.to_csv(out_dir / "design_intent.csv", index=False)
    verdicts.to_csv(out_dir / "verdicts.csv", index=False)

    summary: dict[str, object] = {
        "n_samples": int(len(verdicts)),
        "min_component_area": args.min_area,
        "design_silicon_mismatch_count": int(design["design_silicon_mismatch"].sum()),
        # Sanity check: our component counter must agree with the metrology
        # module's own count on the reference masks.
        "extraction_check_reference_vs_metadata": float(
            (design["n_reference"] == design["metadata_component_count"]).mean()
        ),
        "truth_distribution": verdicts["truth"].value_counts().to_dict(),
        "frontends": {},
    }

    for model in MODELS:
        col = f"{model}_verdict"
        scored = verdicts.dropna(subset=[col])
        if scored.empty:
            continue
        correct = int((scored[col] == scored["truth"]).sum())
        # Split wrong calls by fab consequence: a spurious defect call costs
        # review time, a missed defect ships a bad wafer.
        false_alarm = int(((scored["truth"] == "NOMINAL") & (scored[col] != "NOMINAL")).sum())
        missed = int(((scored["truth"] != "NOMINAL") & (scored[col] == "NOMINAL")).sum())
        miscategorised = int(
            (
                (scored["truth"] != "NOMINAL")
                & (scored[col] != "NOMINAL")
                & (scored[col] != scored["truth"])
            ).sum()
        )
        summary["frontends"][model] = {
            "n": int(len(scored)),
            "correct": correct,
            "accuracy": correct / len(scored),
            "wrong_calls": int(len(scored)) - correct,
            "false_alarm": false_alarm,
            "missed_defect": missed,
            "miscategorised": miscategorised,
            "verdict_distribution": scored[col].value_counts().to_dict(),
        }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"samples: {summary['n_samples']}")
    print(
        "extraction check (reference count vs metrology module): "
        f"{summary['extraction_check_reference_vs_metadata'] * 100:.1f}%"
    )
    print(f"design/silicon mismatches: {summary['design_silicon_mismatch_count']}")
    print(f"truth distribution: {summary['truth_distribution']}\n")
    header = f"{'frontend':<16}{'acc':>7}{'wrong':>7}{'false_alarm':>13}{'missed':>8}{'miscat':>8}"
    print(header)
    for model, s in summary["frontends"].items():
        print(
            f"{model:<16}{s['accuracy']:>7.3f}{s['wrong_calls']:>7}"
            f"{s['false_alarm']:>13}{s['missed_defect']:>8}{s['miscategorised']:>8}"
        )
    print(f"\nartifacts -> {out_dir}")


if __name__ == "__main__":
    main()
