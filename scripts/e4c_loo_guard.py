"""E4c: leave-one-out sensitivity of the cross-frontend disagreement guard.

Answers the reviewer question "does the guard depend on having exactly these
four frontends?" by recomputing the disagreement score d_m under every
3-frontend subset (leave-one-out) and every {segformer, X} pair (minimal
deployable ensemble), then re-scoring failure-detection AUROC.

Reuses cached prediction masks and metrology CSVs; no GPU required.

Outputs:
  output/revision_v4/e4c_loo_guard.json
  output/revision_v4/e4c_loo_guard.csv
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from revision_v4_analysis import (  # noqa: E402
    MODELS, OUT, SPLITS, _f, auroc, iou_of, load_bin, metro_df, names_of,
    resize_like,
)


def load_tau() -> float:
    with open(OUT / "e1b_noise_floor.json") as f:
        nf = json.load(f)
    return float(nf["extreme"]["systematic_1px"]["cd_mean"]["mean"])


def pairwise_iou_table() -> pd.DataFrame:
    """One row per (split, sample): IoU for each unordered frontend pair."""
    rows = []
    pairs = list(combinations(MODELS, 2))
    for split in SPLITS:
        for name in names_of(split):
            mm = {m: load_bin(Path(str(SPLITS[split]["pred"]).format(m=m)) / f"{name}.png")
                  for m in MODELS}
            shape = min(v.shape for v in mm.values())
            mm = {k: resize_like(v, shape) for k, v in mm.items()}
            row = {"split": split, "name": name}
            for a, b in pairs:
                row[f"iou_{a}_{b}"] = iou_of(mm[a], mm[b])
            rows.append(row)
    return pd.DataFrame(rows)


def pair_iou(row: pd.Series, a: str, b: str) -> float:
    key = f"iou_{a}_{b}" if f"iou_{a}_{b}" in row.index else f"iou_{b}_{a}"
    return float(row[key])


def disagreement_scores(pw: pd.DataFrame, ensemble: tuple[str, ...]) -> pd.DataFrame:
    """Per (split, sample, monitored model in ensemble): d = 1 - mean IoU."""
    rows = []
    for _, row in pw.iterrows():
        for m in ensemble:
            others = [o for o in ensemble if o != m]
            d = 1.0 - float(np.mean([pair_iou(row, m, o) for o in others]))
            rows.append({"split": row["split"], "name": row["name"],
                         "model": m, "disagreement": d})
    return pd.DataFrame(rows)


def attach_errors(df: pd.DataFrame) -> pd.DataFrame:
    errs = []
    for split in SPLITS:
        for m in MODELS:
            e = metro_df(split, m)[["name", "abs_err_cd_mean"]].copy()
            e["split"], e["model"] = split, m
            errs.append(e)
    err = pd.concat(errs)
    err["abs_err_cd_mean"] = err["abs_err_cd_mean"].map(_f)
    return df.merge(err, on=["split", "name", "model"], how="left").dropna(
        subset=["abs_err_cd_mean"])


def score(df: pd.DataFrame, tau: float, ensemble: tuple[str, ...]) -> list[dict]:
    out = []
    scopes = [("extreme_all_in_ensemble", (df.split == "extreme")),
              ("pooled_all", df.split.notna())]
    if "segformer" in ensemble:
        scopes.insert(0, ("extreme_segformer",
                          (df.split == "extreme") & (df.model == "segformer")))
    for scope, sel in scopes:
        sub = df[sel]
        lab = sub["abs_err_cd_mean"].to_numpy(float) > tau
        if lab.sum() == 0 or (~lab).sum() == 0:
            continue
        out.append({
            "ensemble": "+".join(ensemble), "scope": scope,
            "n": int(len(sub)), "n_fail": int(lab.sum()),
            "auroc": round(auroc(sub["disagreement"].to_numpy(float), lab), 4),
            "spearman": round(float(stats.spearmanr(
                sub["disagreement"], sub["abs_err_cd_mean"]).statistic), 4),
        })
    return out


def main() -> None:
    tau = load_tau()
    print(f"== E4c: guard ensemble sensitivity (tau={tau:.2f} px) ==")
    pw = pairwise_iou_table()

    ensembles: list[tuple[str, ...]] = [tuple(MODELS)]
    ensembles += [tuple(m for m in MODELS if m != left_out) for left_out in MODELS]
    ensembles += [("segformer", other) for other in MODELS if other != "segformer"]

    results = []
    for ens in ensembles:
        df = attach_errors(disagreement_scores(pw, ens))
        results.extend(score(df, tau, ens))

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / "e4c_loo_guard.csv", index=False)
    json.dump({"tau_px": tau, "results": results},
              open(OUT / "e4c_loo_guard.json", "w"), indent=2)

    for scope in ["extreme_segformer", "extreme_all_in_ensemble", "pooled_all"]:
        sub = res_df[res_df.scope == scope]
        if sub.empty:
            continue
        print(f"\n  scope: {scope}")
        for _, r in sub.iterrows():
            print(f"    {r.ensemble:<45s} n={r.n:<4d} fail={r.n_fail:<3d} "
                  f"AUROC={r.auroc:.3f}  Spearman={r.spearman:.3f}")
    print(f"\n  outputs: {OUT / 'e4c_loo_guard.csv'}")


if __name__ == "__main__":
    main()
