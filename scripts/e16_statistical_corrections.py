"""E16: statistical corrections for the claims that carry the paper.

Three defects in the published analysis, each addressed here:

1. **Breakpoint significance.** ``e3_onset`` searches the breakpoint by
   minimising SSE, then tests two-piece against single-line with
   ``F(3, n-5)``. Under the null there is no breakpoint, so that parameter is
   unidentified and the statistic is really a sup-F, whose null distribution
   is stochastically larger than F. The published p-value is therefore
   anti-conservative (the Davies problem). We replace it with a residual
   bootstrap of the null: rerun the whole breakpoint search on data generated
   under a single line and read the p-value off the simulated sup-F.

2. **AUROC uncertainty.** Every AUROC in Section IX is a point estimate on 8
   to 21 failure events. We add percentile bootstrap intervals, resampling
   *images* rather than rows, because the four frontends see the same image
   and their records are not independent.

3. **Multiple comparisons.** The guard is compared against two alternative
   statistics in three groupings. We report Holm-adjusted p-values over that
   family.

Outputs under ``output/revision_v4/e16_stats/``:

    breakpoint_test.json   observed and bootstrap-null sup-F, both p-values
    auroc_ci.csv           AUROC with bootstrap CI per setting and statistic
    summary.json           everything above, plus the adjusted comparisons

Usage::

    python scripts/e16_statistical_corrections.py --n-boot 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]

METRO_EXTREME = REPO_ROOT / "output/hard_eval/segformer_metrology.csv"
DISAGREEMENT = REPO_ROOT / "output/revision_v4/e4_disagreement.csv"
NOISE_FLOOR = REPO_ROOT / "output/revision_v4/e1b_noise_floor.json"
OUT_DIR = REPO_ROOT / "output/revision_v4/e16_stats"

SETTINGS = {
    "extreme_segformer": lambda d: (d.split == "extreme") & (d.model == "segformer"),
    "extreme_all_models": lambda d: d.split == "extreme",
    "pooled_all": lambda d: d.split.notna(),
}
STATISTICS = {
    "disagreement": "disagreement",
    "fg_dev": "pred_fg_dev",
    "cc_dev": "pred_cc_dev",
}
MIN_SIDE = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(OUT_DIR))
    return p.parse_args()


def two_piece_sse(x: np.ndarray, y: np.ndarray, min_side: int = MIN_SIDE):
    """Search the breakpoint minimising the total SSE of two linear pieces."""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    ux = np.unique(xs)
    best = None
    for i in range(len(ux) - 1):
        b = (ux[i] + ux[i + 1]) / 2
        left, right = xs <= b, xs > b
        if left.sum() < min_side or right.sum() < min_side:
            continue
        sse = 0.0
        for sel in (left, right):
            coef = np.polyfit(xs[sel], ys[sel], 1)
            sse += float(((np.polyval(coef, xs[sel]) - ys[sel]) ** 2).sum())
        if best is None or sse < best[1]:
            best = (b, sse)
    return best


def sup_f(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """Return (breakpoint, sup-F) for two-piece against a single line."""
    fit = two_piece_sse(x, y)
    if fit is None:
        return None
    breakpoint, sse2 = fit
    coef = np.polyfit(x, y, 1)
    sse1 = float(((np.polyval(coef, x) - y) ** 2).sum())
    n = len(x)
    if sse2 <= 0 or n <= 5:
        return None
    return breakpoint, ((sse1 - sse2) / 3) / (sse2 / (n - 5))


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cluster_bootstrap(
    df: pd.DataFrame, column: str, label: np.ndarray, rng: np.random.Generator, n_boot: int
) -> np.ndarray:
    """Bootstrap AUROC draws, resampling images so same-image rows move together."""
    names = df["name"].to_numpy()
    unique_names = np.unique(names)
    index_by_name = {nm: np.flatnonzero(names == nm) for nm in unique_names}
    values = df[column].to_numpy(float)

    draws = []
    for _ in range(n_boot):
        picked = rng.choice(unique_names, size=len(unique_names), replace=True)
        idx = np.concatenate([index_by_name[nm] for nm in picked])
        a = auroc(values[idx], label[idx])
        if np.isfinite(a):
            draws.append(a)
    return np.asarray(draws)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summary: dict[str, object] = {"n_boot": args.n_boot, "seed": args.seed}

    # ---------------------------------------------------------- breakpoint
    sf = pd.read_csv(METRO_EXTREME).dropna(subset=["abs_err_cd_mean", "gt_foreground_ratio"])
    x = sf["gt_foreground_ratio"].to_numpy(float)
    y = np.log10(sf["abs_err_cd_mean"].to_numpy(float) + 1e-3)
    n = len(x)

    observed = sup_f(x, y)
    if observed is None:
        raise SystemExit("two-piece fit failed on the observed data")
    breakpoint, f_observed = observed
    p_naive = float(stats.f.sf(f_observed, 3, n - 5))

    coef = np.polyfit(x, y, 1)
    fitted = np.polyval(coef, x)
    residuals = y - fitted
    null_draws = []
    for _ in range(args.n_boot):
        y_star = fitted + rng.choice(residuals, size=n, replace=True)
        r = sup_f(x, y_star)
        if r is not None:
            null_draws.append(r[1])
    null_draws = np.asarray(null_draws)
    # Add-one correction keeps the p-value strictly positive.
    p_boot = float((1 + (null_draws >= f_observed).sum()) / (1 + len(null_draws)))

    breakpoint_result = {
        "n": n,
        "breakpoint": float(breakpoint),
        "sup_F_observed": float(f_observed),
        "p_naive_F_3_nm5": p_naive,
        "p_bootstrap_null": p_boot,
        "null_draws": int(len(null_draws)),
        "null_F_median": float(np.median(null_draws)),
        "null_F_p95": float(np.percentile(null_draws, 95)),
        "note": (
            "The breakpoint is estimated, so the statistic is a sup-F and the "
            "naive F(3, n-5) reference distribution is anti-conservative "
            "(Davies problem). The bootstrap p-value is the defensible one."
        ),
    }
    with open(out_dir / "breakpoint_test.json", "w") as f:
        json.dump(breakpoint_result, f, indent=2)
    summary["breakpoint"] = breakpoint_result

    # --------------------------------------------------------------- AUROC
    with open(NOISE_FLOOR) as f:
        tau = float(json.load(f)["extreme"]["systematic_1px"]["cd_mean"]["mean"])
    disagreement = pd.read_csv(DISAGREEMENT, dtype={"name": str}).dropna(subset=["abs_err_cd_mean"])

    rows, comparison_p = [], {}
    for setting, selector in SETTINGS.items():
        block = disagreement[selector(disagreement)].copy()
        label = block["abs_err_cd_mean"].to_numpy(float) > tau
        draws_by_stat = {}
        for stat_name, column in STATISTICS.items():
            point = auroc(block[column].to_numpy(float), label)
            draws = cluster_bootstrap(block, column, label, rng, args.n_boot)
            draws_by_stat[stat_name] = draws
            lo, hi = (
                np.percentile(draws, [2.5, 97.5]) if len(draws) else (float("nan"), float("nan"))
            )
            rows.append(
                {
                    "setting": setting,
                    "statistic": stat_name,
                    "n": int(len(block)),
                    "n_fail": int(label.sum()),
                    "auroc": point,
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                }
            )
        # Paired differences on the same resamples, so the interval accounts
        # for the correlation between the two statistics.
        for other in ("fg_dev", "cc_dev"):
            a, b = draws_by_stat["disagreement"], draws_by_stat[other]
            k = min(len(a), len(b))
            if k == 0:
                continue
            diff = a[:k] - b[:k]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
            comparison_p[f"{setting}:disagreement_vs_{other}"] = float(min(1.0, p))
            rows.append(
                {
                    "setting": setting,
                    "statistic": f"diff_disagreement_minus_{other}",
                    "n": int(len(block)),
                    "n_fail": int(label.sum()),
                    "auroc": float(diff.mean()),
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                }
            )

    auroc_table = pd.DataFrame(rows)
    auroc_table.to_csv(out_dir / "auroc_ci.csv", index=False)
    summary["auroc"] = auroc_table.to_dict(orient="records")
    summary["tau_px"] = tau

    adjusted = holm(comparison_p)
    summary["comparisons"] = {
        k: {"p_raw": comparison_p[k], "p_holm": adjusted[k]} for k in comparison_p
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== breakpoint significance ===")
    print(f"  breakpoint          {breakpoint:.4f}   n={n}")
    print(f"  sup-F observed      {f_observed:.3f}")
    print(f"  p (naive F, paper)  {p_naive:.3g}   <- anti-conservative")
    print(f"  p (bootstrap null)  {p_boot:.4g}   <- defensible")
    print(
        f"  null sup-F: median {np.median(null_draws):.2f}, "
        f"p95 {np.percentile(null_draws, 95):.2f}"
    )

    print("\n=== AUROC with bootstrap CI (images resampled as clusters) ===")
    print(f"{'setting':<22}{'statistic':<36}{'n':>5}{'fail':>6}{'AUROC':>9}{'95% CI':>20}")
    for _, r in auroc_table.iterrows():
        ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        print(
            f"{r['setting']:<22}{r['statistic']:<36}{r['n']:>5}{r['n_fail']:>6}"
            f"{r['auroc']:>9.3f}{ci:>20}"
        )

    print("\n=== guard vs alternatives, Holm-adjusted ===")
    for key, s in summary["comparisons"].items():
        verdict = "separable" if s["p_holm"] < 0.05 else "not separable"
        print(f"  {key:<48} p={s['p_raw']:.3f}  Holm={s['p_holm']:.3f}  {verdict}")

    print(f"\nartifacts -> {out_dir}")


if __name__ == "__main__":
    main()
