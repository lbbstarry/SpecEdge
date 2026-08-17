#!/usr/bin/env python
"""Score the guard against the policies it has to beat to be worth deploying.

Every routing number in the paper compares guard-routed SegFormer against
unguarded SegFormer. That comparison cannot establish that the guard is worth
its cost, because the cheapest alternative is to delete the guard and deploy
the fallback frontend on its own. This script scores that alternative on the
same material, on both splits, for both the metrology error and the
layout-referenced design verdict.

The monitored frontend and the threshold are fixed before scoring: SegFormer is
monitored because it is the frontend the qualification study flagged, and the
threshold is the 95th percentile of its in-distribution disagreement. Every
fallback is then scored under that same fixed policy, so no choice made here
uses knowledge of the out-of-window answer.

    python scripts/e22_policy_baselines.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "revision_v4" / "e22_policy"
MODELS = ["unet", "deeplabv3plus", "hrnet", "segformer"]
MONITORED = "segformer"
PERCENTILE = 95.0

SPLIT_SOURCES = {
    "standard": ROOT / "output" / "metrology" / "{m}_test_metrics.csv",
    "extreme": ROOT / "output" / "hard_eval" / "{m}_metrology.csv",
}


def load_cd() -> dict[tuple[str, str], pd.Series]:
    """Per-sample CD error for every frontend on both splits, keyed by name."""
    out = {}
    for split, pattern in SPLIT_SOURCES.items():
        for m in MODELS:
            df = pd.read_csv(str(pattern).format(m=m))
            df = df.dropna(subset=["abs_err_cd_mean"])
            df["name"] = df["name"].astype(str).str.zfill(8)
            out[(split, m)] = df.set_index("name")["abs_err_cd_mean"]
    return out


def _stats(s: pd.Series) -> dict[str, float]:
    return {
        "mean": float(s.mean()),
        "max": float(s.max()),
        "p90": float(np.percentile(s, 90)),
    }


def policy_cd(cd, split: str, flagged: set[str], fallback: str) -> dict:
    """CD error under monitored-only, fallback-only, and guard-routed."""
    monitored, fb = cd[(split, MONITORED)], cd[(split, fallback)]
    common = monitored.index.intersection(fb.index)
    monitored, fb = monitored.loc[common], fb.loc[common]

    flag = pd.Series([n in flagged for n in common], index=common)
    routed = monitored.where(~flag, fb)

    return {
        "n": int(len(common)),
        "flag_rate": float(flag.mean()),
        "always_monitored": _stats(monitored),
        "always_fallback": _stats(fb),
        "routed": _stats(routed),
    }


def policy_verdicts(scored: set[str], flagged: set[str], fallback: str) -> dict:
    """Wrong layout-referenced design calls under the same three policies."""
    v = pd.read_csv(ROOT / "output" / "revision_v4" / "e13_layout_dfm" / "verdicts.csv")
    v["id"] = v["id"].astype(str).str.zfill(8)
    v = v[v["id"].isin(scored)]

    mon, fb = v[f"{MONITORED}_verdict"], v[f"{fallback}_verdict"]
    flag = v["id"].isin(flagged)
    routed = mon.where(~flag, fb)

    return {
        "n": int(len(v)),
        "flag_rate": float(flag.mean()),
        "always_monitored": int((mon != v["truth"]).sum()),
        "always_fallback": int((fb != v["truth"]).sum()),
        "routed": int((routed != v["truth"]).sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cd = load_cd()

    d = pd.read_csv(ROOT / "output" / "revision_v4" / "e4_disagreement.csv")
    d["name"] = d["name"].astype(str).str.zfill(8)
    mon = d[d.model == MONITORED]

    # Calibrated on in-distribution material only, as a fab could before deployment.
    thr = float(np.percentile(mon[mon.split == "standard"]["disagreement"], PERCENTILE))
    flagged = {s: set(g[g.disagreement > thr]["name"]) for s, g in mon.groupby("split")}
    scored_extreme = set(mon[mon.split == "extreme"]["name"])

    result = {
        "monitored": MONITORED,
        "threshold_percentile": PERCENTILE,
        "threshold": thr,
        "note": (
            "always_fallback is the policy of deploying the fallback frontend "
            "alone and deleting the guard. It is the baseline the guard must "
            "beat to justify its cost."
        ),
        "cd_error": {},
        "design_verdicts": {},
    }

    for fallback in [m for m in MODELS if m != MONITORED]:
        result["cd_error"][fallback] = {
            split: policy_cd(cd, split, flagged[split], fallback)
            for split in SPLIT_SOURCES
        }
        result["design_verdicts"][fallback] = policy_verdicts(
            scored_extreme, flagged["extreme"], fallback
        )

    (OUT / "summary.json").write_text(json.dumps(result, indent=2))

    print(f"monitored={MONITORED}   threshold = p{PERCENTILE:.0f} = {thr:.4f}\n")
    head = f"{'':11}{'n':>4}{'flag':>7}{'always-sf':>11}{'always-fb':>11}{'routed':>9}"
    for fallback, per_split in result["cd_error"].items():
        print(f"fallback = {fallback}")
        print("  CD MAE mean" + head[11:])
        for split, r in per_split.items():
            print(f"  {split:9}{r['n']:4}{r['flag_rate']:7.1%}"
                  f"{r['always_monitored']['mean']:11.3f}"
                  f"{r['always_fallback']['mean']:11.3f}{r['routed']['mean']:9.3f}")
        w = result["design_verdicts"][fallback]
        print(f"  wrong design calls  n={w['n']}   always-sf {w['always_monitored']}"
              f"   always-fb {w['always_fallback']}   routed {w['routed']}\n")


if __name__ == "__main__":
    main()
