"""E12: DFM verdict fidelity per frontend, on both splits.

E13 scores topology verdicts against layout design intent, but the layout is
only available for the 65 Extreme samples. This script covers both splits by
scoring the flags that can be derived from a single mask, and asks a narrower
question: when the reference mask says a wafer carries a given DFM risk, does
the frontend's prediction say the same thing?

Flags scored (threshold 0.25 matches ``specedge/metrology.py``):

    necking   necking_score > 0.25    line pinching, an open-circuit risk
    bulging   bulging_score > 0.25    line swelling, a bridging risk
    topology  component_count differs from the reference count

For each frontend and flag we report agreement, false-alarm rate (frontend
raises a risk the reference does not), miss rate (frontend clears a wafer the
reference flags), and Cohen's kappa, which corrects for the fact that most
wafers carry no risk and so a constant "NOMINAL" predictor already scores well.

Outputs under ``output/revision_v4/e12_verdict_fidelity/``:

    per_flag.csv    one row per (split, frontend, flag)
    summary.json    same content plus per-frontend aggregates

Usage::

    python scripts/e12_dfm_verdict_fidelity.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "output/revision_v4/e12_verdict_fidelity"

MODELS = ("unet", "deeplabv3plus", "hrnet", "segformer")

SPLITS = {
    "in_dist": REPO_ROOT / "output/metrology/{m}_test_metrics.csv",
    "extreme": REPO_ROOT / "output/hard_eval/{m}_metrology.csv",
}

# Same cut-off the metrology module uses for necking_candidate / bulging_candidate.
RISK_SCORE_THRESHOLD = 0.25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold", type=float, default=RISK_SCORE_THRESHOLD)
    p.add_argument("--out", default=str(OUT_DIR))
    return p.parse_args()


def cohen_kappa(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Two-class kappa. Returns nan when chance agreement is total."""
    n = len(truth)
    if n == 0:
        return float("nan")
    observed = float((predicted == truth).mean())
    p_pred, p_true = predicted.mean(), truth.mean()
    expected = p_pred * p_true + (1 - p_pred) * (1 - p_true)
    if np.isclose(expected, 1.0):
        return float("nan")
    return float((observed - expected) / (1 - expected))


def score_flag(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    n = len(truth)
    n_true = int(truth.sum())
    false_alarm = int((predicted & ~truth).sum())
    missed = int((~predicted & truth).sum())
    negatives = int((~truth).sum())
    return {
        "n": n,
        "n_flagged_reference": n_true,
        "n_flagged_prediction": int(predicted.sum()),
        "agreement": float((predicted == truth).mean()) if n else float("nan"),
        "false_alarm": false_alarm,
        # Rate over wafers that genuinely carry no risk, i.e. the review burden
        # a fab would absorb for nothing.
        "false_alarm_rate": false_alarm / negatives if negatives else float("nan"),
        "missed": missed,
        "miss_rate": missed / n_true if n_true else float("nan"),
        "kappa": cohen_kappa(predicted, truth),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split, template in SPLITS.items():
        for model in MODELS:
            path = Path(str(template).format(m=model))
            if not path.exists():
                print(f"skip: {path} missing")
                continue
            df = pd.read_csv(path)

            flags = {
                "necking": (
                    df["pred_necking_score"] > args.threshold,
                    df["gt_necking_score"] > args.threshold,
                ),
                "bulging": (
                    df["pred_bulging_score"] > args.threshold,
                    df["gt_bulging_score"] > args.threshold,
                ),
            }
            for flag, (pred, truth) in flags.items():
                rows.append(
                    {
                        "split": split,
                        "model": model,
                        "flag": flag,
                        **score_flag(pred.to_numpy(bool), truth.to_numpy(bool)),
                    }
                )

            # Topology is a deviation count rather than a per-mask flag: the
            # reference defines the component count, so only the prediction can
            # depart from it.
            deviates = (df["pred_component_count"] != df["gt_component_count"]).to_numpy(bool)
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "flag": "topology_deviation",
                    "n": len(df),
                    "n_flagged_reference": 0,
                    "n_flagged_prediction": int(deviates.sum()),
                    "agreement": float((~deviates).mean()),
                    "false_alarm": int(deviates.sum()),
                    "false_alarm_rate": float(deviates.mean()),
                    "missed": 0,
                    "miss_rate": float("nan"),
                    "kappa": float("nan"),
                }
            )

    per_flag = pd.DataFrame(rows)
    per_flag.to_csv(out_dir / "per_flag.csv", index=False)

    summary: dict[str, object] = {
        "threshold": args.threshold,
        "per_flag": per_flag.to_dict(orient="records"),
        "aggregate": {},
    }
    risk_flags = per_flag[per_flag["flag"].isin(["necking", "bulging"])]
    for split in SPLITS:
        block = risk_flags[risk_flags["split"] == split]
        summary["aggregate"][split] = {
            model: {
                "mean_agreement": float(block[block.model == model]["agreement"].mean()),
                "total_false_alarm": int(block[block.model == model]["false_alarm"].sum()),
                "total_missed": int(block[block.model == model]["missed"].sum()),
                "mean_kappa": float(block[block.model == model]["kappa"].mean()),
            }
            for model in MODELS
            if (block.model == model).any()
        }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    for split in SPLITS:
        block = per_flag[per_flag["split"] == split]
        if block.empty:
            continue
        print(f"\n=== {split} ===")
        print(f"{'model':<15}{'flag':<20}{'agree':>8}{'FA':>5}{'FA rate':>9}{'miss':>6}{'kappa':>8}")
        for _, r in block.iterrows():
            kappa = "     n/a" if np.isnan(r["kappa"]) else f"{r['kappa']:>8.3f}"
            print(
                f"{r['model']:<15}{r['flag']:<20}{r['agreement']:>8.3f}"
                f"{r['false_alarm']:>5}{r['false_alarm_rate']:>9.3f}{r['missed']:>6}{kappa}"
            )

    print(f"\nartifacts -> {out_dir}")


if __name__ == "__main__":
    main()
