"""E15: does the runtime guard protect the design-side decision?

Sections VIII-IX score the guard as a detector of metrology error (AUROC on
CD MAE). That is one step removed from what a fab acts on. This script closes
the gap by applying the same guard-triggered routing policy to the
layout-referenced DFM verdicts from E13, and reporting the change in *wrong
design calls* rather than in pixels.

Policy, unchanged from ``scripts/e4d_routing.py``:

    d*       a percentile of the monitored frontend's disagreement on the
             IN-DISTRIBUTION split -- the calibration a fab could perform
             before deployment, using only qualified material
    routing  if d_monitored(I) > d*, take the fallback frontend's verdict for
             image I; otherwise keep the monitored frontend's verdict

No Extreme sample is used to choose d*. The monitored/fallback pairing is
fixed in advance here, but was chosen in the original study after observing
which frontend fails -- a caveat that belongs in the paper, not in the code.

Outputs under ``output/revision_v4/e15_guard_dfm/``:

    routed_verdicts.csv   per-sample verdicts before and after routing
    summary.json          wrong-call counts per policy, plus oracle bounds

Usage::

    python scripts/e15_guard_protects_dfm.py --monitored segformer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

DISAGREEMENT_CSV = REPO_ROOT / "output/revision_v4/e4_disagreement.csv"
VERDICTS_CSV = REPO_ROOT / "output/revision_v4/e13_layout_dfm/verdicts.csv"
OUT_DIR = REPO_ROOT / "output/revision_v4/e15_guard_dfm"

MODELS = ("unet", "deeplabv3plus", "hrnet", "segformer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--monitored", default="segformer")
    p.add_argument(
        "--percentiles",
        type=int,
        nargs="+",
        default=[80, 90, 95],
        help="in-distribution percentiles of d to use as d*",
    )
    p.add_argument("--out", default=str(OUT_DIR))
    return p.parse_args()


def wrong_calls(verdicts: pd.Series, truth: pd.Series) -> dict[str, int]:
    return {
        "wrong": int((verdicts != truth).sum()),
        "false_alarm": int(((truth == "NOMINAL") & (verdicts != "NOMINAL")).sum()),
        "missed": int(((truth != "NOMINAL") & (verdicts == "NOMINAL")).sum()),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    disagreement = pd.read_csv(DISAGREEMENT_CSV, dtype={"name": str})
    verdicts = pd.read_csv(VERDICTS_CSV, dtype={"id": str}).rename(columns={"id": "name"})

    monitored = args.monitored
    fallbacks = [m for m in MODELS if m != monitored]

    # d* comes from the in-distribution split only; Extreme is never consulted.
    in_dist = disagreement[(disagreement.split == "standard") & (disagreement.model == monitored)]
    thresholds = {p: float(np.percentile(in_dist["disagreement"], p)) for p in args.percentiles}

    extreme = disagreement[
        (disagreement.split == "extreme") & (disagreement.model == monitored)
    ][["name", "disagreement"]]
    df = verdicts.merge(extreme, on="name", how="inner")
    if df.empty:
        raise SystemExit("no Extreme samples joined between disagreement and verdict tables")

    truth = df["truth"]
    baseline = wrong_calls(df[f"{monitored}_verdict"], truth)

    summary: dict[str, object] = {
        "monitored": monitored,
        "n": int(len(df)),
        "calibration_percentiles": thresholds,
        "baseline_monitored_only": baseline,
        "fallback_only": {f: wrong_calls(df[f"{f}_verdict"], truth) for f in fallbacks},
        "routed": {},
    }

    # Oracle bound: wrong calls remaining if an omniscient policy always picked
    # whichever frontend happens to be right for each sample.
    per_sample_correct = pd.DataFrame({m: (df[f"{m}_verdict"] == truth) for m in MODELS})
    summary["oracle_best_per_sample_wrong"] = int((~per_sample_correct.any(axis=1)).sum())

    for pct, d_star in thresholds.items():
        flagged = df["disagreement"] > d_star
        for fallback in fallbacks:
            routed = df[f"{monitored}_verdict"].where(~flagged, df[f"{fallback}_verdict"])
            stats = wrong_calls(routed, truth)
            stats.update(
                {
                    "d_star": d_star,
                    "flag_rate": float(flagged.mean()),
                    "n_flagged": int(flagged.sum()),
                    "wrong_before": baseline["wrong"],
                    "wrong_reduction": baseline["wrong"] - stats["wrong"],
                }
            )
            summary["routed"][f"p{pct}_{fallback}"] = stats
            df[f"routed_p{pct}_{fallback}"] = routed

    df.to_csv(out_dir / "routed_verdicts.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"monitored = {monitored}   n = {len(df)}")
    print(
        f"baseline (monitored only): wrong {baseline['wrong']}  "
        f"(false alarm {baseline['false_alarm']}, missed {baseline['missed']})"
    )
    print("fallback alone:")
    for name, s in summary["fallback_only"].items():
        print(f"  {name:<15} wrong {s['wrong']:>3}  (FA {s['false_alarm']}, missed {s['missed']})")
    print(f"oracle (best frontend per sample): wrong {summary['oracle_best_per_sample_wrong']}\n")

    print(f"{'policy':<24}{'d*':>9}{'flag%':>8}{'wrong':>7}{'delta':>7}{'FA':>5}{'miss':>6}")
    for key, s in summary["routed"].items():
        print(
            f"{key:<24}{s['d_star']:>9.4f}{s['flag_rate'] * 100:>7.1f}%"
            f"{s['wrong']:>7}{-s['wrong_reduction']:>+7}{s['false_alarm']:>5}{s['missed']:>6}"
        )
    print(f"\nartifacts -> {out_dir}")


if __name__ == "__main__":
    main()
