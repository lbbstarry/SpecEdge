#!/usr/bin/env python
"""Demonstrate the metric-measurand decoupling under controlled perturbation.

Section VII-C derives that overlap reads one functional of the boundary
displacement field, its L1 norm attenuated by L/A, while CD reads a signed mean
of the same field and LER/LWR read a standard deviation after detrending. If
that is right, displacement fields with *identical* mean |delta| must produce
identical IoU while their CD and roughness errors range over the full predicted
interval.

This script constructs exactly that comparison on the reference masks. Three
perturbation families, all carrying mean |delta| = c by construction:

    constant      delta = +c everywhere      predict dCD = 2c, LER 3s = 0
    rademacher    delta in {-c, +c}, iid     predict dCD ~ 0,  LER 3s = 3c
    gaussian      zero-mean, E|delta| = c    predict dCD ~ 0,  LER 3s = 3c*sqrt(pi/2)

Overlap cannot separate the three; the metrology quantities separate them
completely. The second prediction, that the attenuation scales with L/A, is
checked by regressing the observed 1-IoU on (L/A)*c.

    python scripts/e23_mechanism_synthetic.py                 # all masks, c = 1,2,3
    python scripts/e23_mechanism_synthetic.py --limit 40      # quick check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "output" / "revision_v4" / "e23_mechanism"

from specedge.metrology import (  # noqa: E402
    estimate_dominant_orientation,
    evaluate_metrology_pair,
)

MASK_DIRS = [
    ROOT / "dataset" / "litho" / "masks" / "train",
    ROOT / "dataset" / "litho" / "masks" / "val",
    ROOT / "dataset" / "litho" / "masks" / "test",
]
ARMS = ("constant", "rademacher", "gaussian")


def runs_along(column: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive [start, end] index pairs of the True runs in a 1-D array."""
    padded = np.concatenate(([False], column, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2])]


def displacements(arm: str, c: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Displacement field with mean |delta| = c, for every arm."""
    if arm == "constant":
        return np.full(n, c)
    if arm == "rademacher":
        return c * rng.choice([-1.0, 1.0], size=n)
    if arm == "gaussian":
        # E|X| = sigma*sqrt(2/pi) = c  =>  sigma = c*sqrt(pi/2)
        return rng.normal(0.0, c * np.sqrt(np.pi / 2.0), size=n)
    raise ValueError(arm)


def perturb(mask: np.ndarray, arm: str, c: float, rng: np.random.Generator) -> np.ndarray:
    """Displace both edges of every line by delta, outward positive.

    Scans across the line direction and moves each run's two endpoints
    independently, so the perturbation acts on the edge profiles the metrology
    extractor reads rather than on the area.
    """
    out = np.zeros_like(mask)
    n_scan = mask.shape[1]
    top = displacements(arm, c, n_scan, rng)
    bot = displacements(arm, c, n_scan, rng)
    limit = mask.shape[0] - 1
    for j in range(n_scan):
        dt, db = int(round(top[j])), int(round(bot[j]))
        for a, b in runs_along(mask[:, j]):
            a2 = min(max(a - dt, 0), limit)
            b2 = min(max(b + db, 0), limit)
            if b2 >= a2:
                out[a2:b2 + 1, j] = True
    return out


def boundary_over_area(mask: np.ndarray) -> float:
    """L/A, the attenuation factor of Eq. (1), measured on the raster."""
    area = int(mask.sum())
    if area == 0:
        return float("nan")
    inner = (
        mask
        & np.roll(mask, 1, 0) & np.roll(mask, -1, 0)
        & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
    )
    return float((area - int(inner.sum())) / area)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N masks")
    ap.add_argument("--c", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for d in MASK_DIRS if d.exists() for p in d.glob("*.png"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no masks under {[str(d) for d in MASK_DIRS]}")

    rng = np.random.default_rng(args.seed)
    rows = []
    skipped = 0
    for i, path in enumerate(paths, 1):
        ref = np.array(Image.open(path)) > 127
        u8 = ref.astype(np.uint8)
        orientation = estimate_dominant_orientation(u8)
        if orientation == "complex":
            skipped += 1
            continue
        # Scan across the line direction: columns for horizontal lines.
        work = ref if orientation == "horizontal" else ref.T
        la = boundary_over_area(work)

        for c in args.c:
            for arm in ARMS:
                pert = perturb(work, arm, c, rng)
                back = pert if orientation == "horizontal" else pert.T
                m = evaluate_metrology_pair(back.astype(np.uint8), u8)
                inter = int((back & ref).sum())
                union = int((back | ref).sum())
                rows.append({
                    "name": path.stem,
                    "arm": arm,
                    "c_px": c,
                    "L_over_A": la,
                    "iou": inter / union if union else np.nan,
                    "abs_err_cd_mean": m.get("abs_err_cd_mean", np.nan),
                    # Absolute levels, because roughness adds in quadrature: the
                    # perturbed value is sqrt(reference^2 + injected^2), so the
                    # error alone understates the injected component.
                    "cd_ref": m.get("gt_cd_mean", np.nan),
                    "cd_pert": m.get("pred_cd_mean", np.nan),
                    "ler_ref": m.get("gt_ler_mean_3sigma", np.nan),
                    "ler_pert": m.get("pred_ler_mean_3sigma", np.nan),
                    "lwr_ref": m.get("gt_lwr_3sigma", np.nan),
                    "lwr_pert": m.get("pred_lwr_3sigma", np.nan),
                })
        if i % 50 == 0:
            print(f"  {i}/{len(paths)} masks", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_mask.csv", index=False)

    summary = {
        "n_masks": int(df.name.nunique()),
        "n_skipped_complex": skipped,
        "c_values": args.c,
        "seed": args.seed,
        "note": (
            "All three arms carry mean |delta| = c by construction, so Eq. (1) "
            "predicts equal IoU. CD reads a signed mean and LER a standard "
            "deviation, so those separate the arms."
        ),
        "by_c": {},
        "attenuation": {},
    }

    for c, g in df.groupby("c_px"):
        per_arm = {}
        for arm, ga in g.groupby("arm"):
            injected = 0.0 if arm == "constant" else (
                3.0 * c if arm == "rademacher" else 3.0 * c * float(np.sqrt(np.pi / 2))
            )
            ref_ler = float(ga.ler_ref.mean())
            per_arm[arm] = {
                "iou_mean": float(ga.iou.mean()),
                "iou_sd": float(ga.iou.std()),
                "cd_err_mean": float(ga.abs_err_cd_mean.mean()),
                "ler_ref_mean": ref_ler,
                "ler_pert_mean": float(ga.ler_pert.mean()),
                "lwr_pert_mean": float(ga.lwr_pert.mean()),
                "ler_injected_3sigma": injected,
                "ler_pert_predicted_quadrature": float(np.hypot(ref_ler, injected)),
            }
        ious = [per_arm[a]["iou_mean"] for a in ARMS if a in per_arm]
        cds = [per_arm[a]["cd_err_mean"] for a in ARMS if a in per_arm]
        lers = [per_arm[a]["ler_pert_mean"] for a in ARMS if a in per_arm]
        summary["by_c"][str(c)] = {
            "arms": per_arm,
            "iou_spread_across_arms": float(max(ious) - min(ious)),
            "cd_spread_across_arms": float(max(cds) - min(cds)),
            "ler_spread_across_arms": float(max(lers) - min(lers)),
            "predicted": {
                "constant_cd": 2 * c,
                "rademacher_ler_3sigma": 3 * c,
                "gaussian_ler_3sigma": 3 * c * float(np.sqrt(np.pi / 2)),
            },
        }

    # Second prediction: 1 - IoU = (L/A) * mean|delta|, so slope 1 through 0.
    sub = df[df.arm == "constant"].dropna(subset=["iou", "L_over_A"])
    x = (sub.L_over_A * sub.c_px).values
    y = (1.0 - sub.iou).values
    if len(x) > 2:
        slope = float(np.linalg.lstsq(x[:, None], y, rcond=None)[0][0])
        r = float(np.corrcoef(x, y)[0, 1])
        summary["attenuation"] = {
            "model": "1 - IoU ~ (L/A) * c, no intercept",
            "slope": slope, "pearson_r": r, "n": int(len(x)),
        }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{summary['n_masks']} masks, {skipped} skipped as complex\n")
    for c, blk in summary["by_c"].items():
        cf = float(c)
        print(f"c = {c} px   (all arms have mean|delta| = {c})")
        print(f"    {'arm':12}{'IoU':>9}{'CD err':>9}{'LER obs':>9}{'LER pred':>10}"
              f"   predicted CD")
        for arm in ARMS:
            if arm not in blk["arms"]:
                continue
            a = blk["arms"][arm]
            pred = "2c = %.0f" % (2 * cf) if arm == "constant" else "~0"
            print(f"    {arm:12}{a['iou_mean']:9.4f}{a['cd_err_mean']:9.3f}"
                  f"{a['ler_pert_mean']:9.3f}{a['ler_pert_predicted_quadrature']:10.3f}"
                  f"   {pred}")
        print(f"    IoU spread across arms {blk['iou_spread_across_arms']:.4f}"
              f"   CD spread {blk['cd_spread_across_arms']:.3f}"
              f"   LER spread {blk['ler_spread_across_arms']:.3f}\n")
    if summary["attenuation"]:
        a = summary["attenuation"]
        print(f"attenuation check: slope {a['slope']:.3f} (predicted 1.0), "
              f"r = {a['pearson_r']:.3f}, n = {a['n']}")


if __name__ == "__main__":
    main()
