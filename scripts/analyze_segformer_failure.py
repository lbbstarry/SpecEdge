"""Analyze SegFormer failure on external hard set.

Three sub-analyses (no GPU required, all from cached per-sample CSV/JSON):
  1. Spearman rank correlation IoU vs CD/LWR/LER MAE on standard test and hard set
  2. Sparse-pattern ablation on standard test (split by GT foreground ratio)
  3. Foreground ratio / pattern complexity distribution standard vs hard
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
STD_METROLOGY = ROOT / "output/metrology"
HARD_METROLOGY = ROOT / "output/hard_eval"
STD_BASELINE = ROOT / "output/baselines"
OUT = ROOT / "output/segformer_failure_analysis"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["unet", "deeplabv3plus", "hrnet", "segformer"]


def load_per_sample_iou(model: str, split: str) -> pd.DataFrame:
    """Load per-sample IoU for the given model and split (standard / hard)."""
    if split == "standard":
        path = STD_BASELINE / model / "eval_test.json"
    else:
        path = HARD_METROLOGY / f"{model}_eval.json"
    with open(path) as f:
        data = json.load(f)
    rows = []
    for s in data["per_sample"]:
        rows.append({"name": str(s["name"]), "iou": s["iou"]})
    return pd.DataFrame(rows)


def load_metrology(model: str, split: str) -> pd.DataFrame:
    """Load per-sample metrology CSV. Force name column to str so leading zeros survive."""
    if split == "standard":
        path = STD_METROLOGY / f"{model}_test_metrics.csv"
    else:
        path = HARD_METROLOGY / f"{model}_metrology.csv"
    df = pd.read_csv(path, dtype={"name": str})
    return df


def analysis_1_rank_correlation() -> pd.DataFrame:
    """Spearman rank corr between IoU and CD/LWR/LER MAE per model per split."""
    rows = []
    for split in ["standard", "hard"]:
        for model in MODELS:
            iou_df = load_per_sample_iou(model, split)
            metro = load_metrology(model, split)
            merged = iou_df.merge(metro, on="name", how="inner")
            if len(merged) < 5:
                print(f"WARN: {model}/{split} only {len(merged)} samples")
                continue
            for metric in ["abs_err_cd_mean", "abs_err_lwr_3sigma", "abs_err_ler_mean_3sigma"]:
                sub = merged[["iou", metric]].dropna()
                if len(sub) < 5:
                    continue
                rho, p = spearmanr(sub["iou"], sub[metric])
                rows.append({
                    "split": split,
                    "model": model,
                    "metric": metric.replace("abs_err_", ""),
                    "spearman_rho": rho,
                    "p_value": p,
                    "n": len(sub),
                })
    return pd.DataFrame(rows)


def analysis_2_sparse_ablation() -> dict:
    """Split standard test by GT foreground ratio (lowest 1/3 = sparse).

    If SegFormer is also worse on sparse standard test samples, hypothesis
    "transformer struggles with sparse-foreground patterns" is supported and
    its hard-set failure is NOT external-set-specific.
    """
    # Pull GT foreground ratio from any per-model standard CSV (gt_ columns are identical).
    gt = load_metrology("unet", "standard")[["name", "gt_status", "gt_foreground_ratio"]]
    gt = gt[gt["gt_status"] == "ok"].copy()

    thresh = gt["gt_foreground_ratio"].quantile(1.0 / 3.0)
    sparse_names = set(gt[gt["gt_foreground_ratio"] <= thresh]["name"])
    dense_names = set(gt[gt["gt_foreground_ratio"] > thresh]["name"])

    result = {
        "fg_ratio_threshold": float(thresh),
        "n_sparse": len(sparse_names),
        "n_dense": len(dense_names),
        "per_model": {},
    }

    for model in MODELS:
        metro = load_metrology(model, "standard")
        sparse = metro[metro["name"].isin(sparse_names)]
        dense = metro[metro["name"].isin(dense_names)]
        result["per_model"][model] = {
            "sparse_cd_mae": float(sparse["abs_err_cd_mean"].mean()),
            "sparse_lwr_mae": float(sparse["abs_err_lwr_3sigma"].mean()),
            "sparse_ler_mae": float(sparse["abs_err_ler_mean_3sigma"].mean()),
            "dense_cd_mae": float(dense["abs_err_cd_mean"].mean()),
            "dense_lwr_mae": float(dense["abs_err_lwr_3sigma"].mean()),
            "dense_ler_mae": float(dense["abs_err_ler_mean_3sigma"].mean()),
            "sparse_over_dense_cd": float(sparse["abs_err_cd_mean"].mean() / max(dense["abs_err_cd_mean"].mean(), 1e-9)),
        }
    return result


def analysis_2b_segformer_failure_profile() -> dict:
    """What makes a hard-set sample fail for SegFormer?

    Rank hard samples by SegFormer CD error. Look at top-10 worst cases:
    foreground ratio, component count, GT linewidth, GT roughness.
    Compare with bottom-10 (best) cases. If a clear feature separates them,
    that is the actual failure trigger.
    """
    seg = load_metrology("segformer", "hard")
    seg = seg.dropna(subset=["abs_err_cd_mean"]).copy()
    seg_sorted = seg.sort_values("abs_err_cd_mean", ascending=False)

    worst10 = seg_sorted.head(10)
    best10 = seg_sorted.tail(10)

    def summarize(sub: pd.DataFrame) -> dict:
        return {
            "n": int(len(sub)),
            "cd_mae_mean": float(sub["abs_err_cd_mean"].mean()),
            "cd_mae_max": float(sub["abs_err_cd_mean"].max()),
            "gt_fg_ratio_mean": float(sub["gt_foreground_ratio"].mean()),
            "gt_fg_ratio_median": float(sub["gt_foreground_ratio"].median()),
            "gt_cd_mean": float(sub["gt_cd_mean"].mean()),
            "gt_lwr_3sigma_mean": float(sub["gt_lwr_3sigma"].mean()),
            "gt_ler_mean_3sigma_mean": float(sub["gt_ler_mean_3sigma"].mean()),
            "gt_component_count_mean": float(sub["gt_component_count"].mean()),
            "gt_profile_count_mean": float(sub["gt_profile_count"].mean()),
            "pred_fg_ratio_mean": float(sub["pred_foreground_ratio"].mean()),
            "pred_component_count_mean": float(sub["pred_component_count"].mean()),
            "pred_minus_gt_fg_ratio_mean": float(
                (sub["pred_foreground_ratio"] - sub["gt_foreground_ratio"]).mean()
            ),
            "pred_minus_gt_components_mean": float(
                (sub["pred_component_count"] - sub["gt_component_count"]).mean()
            ),
        }

    return {
        "worst10": summarize(worst10),
        "best10": summarize(best10),
        "worst10_names": worst10["name"].tolist(),
        "best10_names": best10["name"].tolist(),
    }


def analysis_3_distribution() -> dict:
    """Compare GT foreground ratio and component count distributions."""
    ref_std = load_metrology("unet", "standard")
    ref_hard = load_metrology("unet", "hard")

    return {
        "standard": {
            "fg_ratio_mean": float(ref_std["gt_foreground_ratio"].mean()),
            "fg_ratio_median": float(ref_std["gt_foreground_ratio"].median()),
            "fg_ratio_min": float(ref_std["gt_foreground_ratio"].min()),
            "component_count_mean": float(ref_std["gt_component_count"].mean()),
            "n": int(len(ref_std)),
        },
        "hard": {
            "fg_ratio_mean": float(ref_hard["gt_foreground_ratio"].mean()),
            "fg_ratio_median": float(ref_hard["gt_foreground_ratio"].median()),
            "fg_ratio_min": float(ref_hard["gt_foreground_ratio"].min()),
            "component_count_mean": float(ref_hard["gt_component_count"].mean()),
            "n": int(len(ref_hard)),
        },
    }


def main() -> None:
    print("=" * 70)
    print("ANALYSIS 1: Spearman rank correlation (IoU vs metrology MAE)")
    print("=" * 70)
    df1 = analysis_1_rank_correlation()
    df1_pivot = df1.pivot_table(
        index=["split", "model"], columns="metric",
        values="spearman_rho",
    ).round(3)
    print(df1_pivot)
    df1.to_csv(OUT / "rank_correlation.csv", index=False)
    df1_pivot.to_csv(OUT / "rank_correlation_pivot.csv")
    print(f"\nSaved: {OUT}/rank_correlation.csv")

    print()
    print("=" * 70)
    print("ANALYSIS 2: Sparse-pattern ablation on standard test")
    print("=" * 70)
    r2 = analysis_2_sparse_ablation()
    print(f"FG ratio threshold (sparse = bottom 1/3): {r2['fg_ratio_threshold']:.4f}")
    print(f"n_sparse={r2['n_sparse']}, n_dense={r2['n_dense']}\n")
    print(f"{'Model':<14} {'sparse CD':>10} {'dense CD':>10} {'sparse/dense':>14}")
    for m in MODELS:
        d = r2["per_model"][m]
        print(f"{m:<14} {d['sparse_cd_mae']:>10.4f} {d['dense_cd_mae']:>10.4f} {d['sparse_over_dense_cd']:>14.2f}x")
    with open(OUT / "sparse_ablation.json", "w") as f:
        json.dump(r2, f, indent=2)
    print(f"\nSaved: {OUT}/sparse_ablation.json")

    print()
    print("=" * 70)
    print("ANALYSIS 2b: SegFormer failure profile on hard set")
    print("=" * 70)
    r2b = analysis_2b_segformer_failure_profile()
    print(f"WORST 10 samples (by SegFormer CD MAE): {r2b['worst10_names']}")
    print(f"  CD MAE mean = {r2b['worst10']['cd_mae_mean']:.3f} px (max {r2b['worst10']['cd_mae_max']:.3f})")
    print(f"  GT fg_ratio mean = {r2b['worst10']['gt_fg_ratio_mean']:.4f}, median = {r2b['worst10']['gt_fg_ratio_median']:.4f}")
    print(f"  GT components mean = {r2b['worst10']['gt_component_count_mean']:.2f}")
    print(f"  GT profile_count mean = {r2b['worst10']['gt_profile_count_mean']:.2f}")
    print(f"  GT CD mean = {r2b['worst10']['gt_cd_mean']:.2f}")
    print(f"  pred - gt fg_ratio = {r2b['worst10']['pred_minus_gt_fg_ratio_mean']:+.4f}")
    print(f"  pred - gt components = {r2b['worst10']['pred_minus_gt_components_mean']:+.2f}")
    print()
    print(f"BEST 10 samples: {r2b['best10_names']}")
    print(f"  CD MAE mean = {r2b['best10']['cd_mae_mean']:.3f} px")
    print(f"  GT fg_ratio mean = {r2b['best10']['gt_fg_ratio_mean']:.4f}, median = {r2b['best10']['gt_fg_ratio_median']:.4f}")
    print(f"  GT components mean = {r2b['best10']['gt_component_count_mean']:.2f}")
    print(f"  GT CD mean = {r2b['best10']['gt_cd_mean']:.2f}")
    print(f"  pred - gt fg_ratio = {r2b['best10']['pred_minus_gt_fg_ratio_mean']:+.4f}")
    print(f"  pred - gt components = {r2b['best10']['pred_minus_gt_components_mean']:+.2f}")
    with open(OUT / "segformer_failure_profile.json", "w") as f:
        json.dump(r2b, f, indent=2)
    print(f"\nSaved: {OUT}/segformer_failure_profile.json")

    print()
    print("=" * 70)
    print("ANALYSIS 3: Distribution comparison")
    print("=" * 70)
    r3 = analysis_3_distribution()
    for split in ["standard", "hard"]:
        d = r3[split]
        print(f"{split:<10} n={d['n']:>3}  fg_ratio mean={d['fg_ratio_mean']:.4f} median={d['fg_ratio_median']:.4f} min={d['fg_ratio_min']:.4f}  components={d['component_count_mean']:.2f}")
    with open(OUT / "distribution.json", "w") as f:
        json.dump(r3, f, indent=2)
    print(f"\nSaved: {OUT}/distribution.json")


if __name__ == "__main__":
    main()
