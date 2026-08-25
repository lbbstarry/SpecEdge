"""E25: the empirical premises of the budgeted-routing formulation.

The formulation's suboptimality result bites only if the reference-free
statistics are not monotone transforms of one another, and its budget
argument needs the component-count statistic's coarseness. This records
both: pairwise Spearman correlations among the three statistics (Extreme
and pooled), and the tie structure of the component-count deviation.

Run manually:
    python scripts/e25_formulation_stats.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "output/revision_v4/e4_disagreement.csv"
OUT = ROOT / "output/revision_v4/e25_formulation_stats/summary.json"


def pairwise(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)),
        "spearman_d_fg": round(float(
            stats.spearmanr(df.disagreement, df.pred_fg_dev)[0]), 4),
        "spearman_d_cc": round(float(
            stats.spearmanr(df.disagreement, df.pred_cc_dev)[0]), 4),
        "spearman_fg_cc": round(float(
            stats.spearmanr(df.pred_fg_dev, df.pred_cc_dev)[0]), 4),
    }


def main() -> None:
    df = pd.read_csv(SRC).dropna(
        subset=["disagreement", "pred_fg_dev", "pred_cc_dev"])
    ext = df[df.split == "extreme"]
    cc = ext.pred_cc_dev
    res = {
        "extreme": pairwise(ext),
        "pooled": pairwise(df),
        "cc_dev_extreme": {
            "n": int(len(cc)),
            "n_distinct": int(cc.nunique()),
            "max_tie_frac": round(float(
                cc.value_counts().iloc[0] / len(cc)), 4),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
