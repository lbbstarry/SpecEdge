"""E21: recompute the IoU-metrology decoupling on the cross-validation folds.

The headline decoupling result is measured on the single 60-image
in-distribution test split, which gives 240 frontend-image records and leaves
the narrowest IoU bin holding 42 of them. Every image already appears exactly
once as held-out data across the cross-validation folds of E18, so the same
analysis can be run on 588 images and 2352 records without any new training and
without leakage: each record is scored by a model that did not see that image.

For each narrow IoU bin we report the conditional spread of the metrology
error, the p95/p5 ratio, which asks how much measurement error varies among
frontend-image pairs whose overlap scores are effectively equivalent.

Outputs under ``output/revision_v4/e21_decoupling_cv/``:

    records.csv   per frontend-image IoU and metrology errors
    summary.json  per-bin spread, and the Spearman correlations for context

Usage::

    python scripts/e21_decoupling_cv.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO_ROOT = Path(__file__).resolve().parents[1]

EVAL_JSON = REPO_ROOT / "output/cv/fold{k}/{m}/eval_test_fold.json"
METRO_CSV = REPO_ROOT / "output/cv/fold{k}/{m}/metrology_test.csv"
OUT_DIR = REPO_ROOT / "output/revision_v4/e21_decoupling_cv"

MODELS = ("unet", "deeplabv3plus", "hrnet", "segformer")
METRICS = ("abs_err_cd_mean", "abs_err_lwr_3sigma", "abs_err_ler_mean_3sigma")
# Same bin edges as the single-split analysis, so the two are comparable.
BIN_EDGES = [0.98, 0.985, 0.99, 0.995, 1.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--out", default=str(OUT_DIR))
    return p.parse_args()


def normalise(name: object) -> str:
    s = str(name)
    return s.lstrip("0") or "0"


def assemble(n_folds: int) -> pd.DataFrame:
    rows = []
    for k in range(n_folds):
        for m in MODELS:
            ev = Path(str(EVAL_JSON).format(k=k, m=m))
            mt = Path(str(METRO_CSV).format(k=k, m=m))
            if not ev.exists() or not mt.exists():
                continue
            per = json.load(open(ev))["per_sample"]
            iou = {normalise(r["name"]): float(r["iou"]) for r in per}
            metro = pd.read_csv(mt, dtype={"name": str})
            metro["key"] = metro["name"].map(normalise)
            for _, r in metro.iterrows():
                if r["key"] not in iou:
                    continue
                rec = {"fold": k, "model": m, "name": r["key"], "iou": iou[r["key"]]}
                for metric in METRICS:
                    v = r.get(metric)
                    rec[metric] = float(v) if pd.notna(v) else np.nan
                rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = assemble(args.n_folds)
    if df.empty:
        raise SystemExit("no cross-validation records found; run E18 first")
    df.to_csv(out_dir / "records.csv", index=False)

    n_images = df["name"].nunique()
    print(f"records {len(df)} over {n_images} distinct images and {df.model.nunique()} frontends")
    print(f"IoU range {df.iou.min():.4f}--{df.iou.max():.4f}\n")

    summary: dict[str, object] = {
        "n_records": int(len(df)),
        "n_images": int(n_images),
        "bin_edges": BIN_EDGES,
        "bins": [],
        "spearman": {},
    }

    for metric in METRICS:
        sub = df.dropna(subset=[metric])
        if len(sub) > 10:
            r = sps.spearmanr(sub["iou"], sub[metric])
            summary["spearman"][metric] = {
                "rho": float(r.statistic), "p": float(r.pvalue), "n": int(len(sub)),
            }

    print(f"{'IoU bin':<18}{'n':>6}{'metric':<26}{'p5':>10}{'p95':>10}{'spread':>9}")
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        block = df[(df.iou >= lo) & (df.iou < hi)]
        if len(block) < 8:
            continue
        entry: dict[str, object] = {"bin": f"[{lo}, {hi})", "n": int(len(block)), "metrics": {}}
        for metric in METRICS:
            vals = block[metric].dropna()
            vals = vals[vals > 0]
            if len(vals) < 8:
                continue
            p5, p50, p95 = np.percentile(vals, [5, 50, 95])
            spread = float(p95 / p5) if p5 > 0 else float("nan")
            entry["metrics"][metric] = {
                "n": int(len(vals)), "p5": float(p5), "p50": float(p50),
                "p95": float(p95), "spread_p95_over_p5": spread,
            }
            print(f"{entry['bin']:<18}{len(vals):>6}{metric:<26}"
                  f"{p5:>10.4f}{p95:>10.4f}{spread:>9.1f}")
        summary["bins"].append(entry)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSpearman(IoU, metrology error) over all records:")
    for metric, s in summary["spearman"].items():
        print(f"  {metric:<26} rho={s['rho']:+.3f}  p={s['p']:.2g}  n={s['n']}")
    print(f"\nartifacts -> {out_dir}")


if __name__ == "__main__":
    main()
