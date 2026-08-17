"""E19: a calibrated conditional-risk model, and how far its calibration
survives the move out of window.

Section X characterises the optimal routing policy as a pointwise argmin of
conditional risk plus Lagrangian-weighted cost. That characterisation presumes
the conditional risk

    r_0(phi) = P(the monitored frontend's DFM flags disagree with the
                 reference | runtime features phi)

is known. It is not; it has to be estimated, and a policy built on a
miscalibrated estimate inherits that error. This script estimates it, then asks
the question the deployment setting forces: the model is fitted on qualified
in-window material but applied out of window -- does its calibration survive?

Protocol, following the no-leakage discipline used for the guard threshold:

* **Features are runtime-only.** Each record carries the predicted metrology
  quantities of the monitored frontend plus three cross-frontend statistics
  (IoU disagreement, foreground-ratio deviation, component-count deviation).
  No ``gt_*`` column enters the feature matrix.
* **Label is a design-relevant error**, not a pixel one: the frontend's
  necking/bulging risk flags disagree with the reference-derived flags. Unlike
  the layout-referenced verdict of E13, this is computable on both splits.
* **In-distribution records come from the cross-validation folds**, so every
  image is scored by a model that did not train on it: 588 images rather than
  the single 60-image split.
* **Cross-fitting.** Risk estimates for a fold come from a model fitted on the
  other folds, so no record informs its own estimate.
* **Calibration** is isotonic, fitted inside the cross-fitting loop.

Reported: Brier score with its Murphy decomposition (reliability, resolution,
uncertainty), expected calibration error, and a reliability table -- once
in-distribution and once on Extreme, where the fitted model is applied
unchanged.

Outputs under ``output/revision_v4/e19_risk_model/``:

    records.csv       assembled feature/label table for both splits
    reliability.csv   binned reliability data for both splits
    calibration.json  Brier decomposition, ECE, AUROC, drift summary

Usage::

    python scripts/e19_risk_model.py --monitored segformer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats as sps

REPO_ROOT = Path(__file__).resolve().parents[1]

FOLDS_JSON = REPO_ROOT / "dataset/litho_cv/folds.json"
CV_METRO = REPO_ROOT / "output/cv/fold{k}/{m}/metrology_test.csv"
CV_PRED = REPO_ROOT / "output/cv/fold{k}/{m}/preds_test/masks"
EXTREME_METRO = REPO_ROOT / "output/hard_eval/{m}_metrology.csv"
EXTREME_PRED = REPO_ROOT / "output/hard_eval/{m}/preds/masks"
OUT_DIR = REPO_ROOT / "output/revision_v4/e19_risk_model"

MODELS = ("unet", "deeplabv3plus", "hrnet", "segformer")
RISK_THRESHOLD = 0.25  # matches specedge/metrology.py
N_BINS = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--monitored", default="segformer")
    p.add_argument("--n-folds", type=int, default=5, help="cross-fitting folds")
    p.add_argument("--feature-set", choices=("cross_frontend", "all"),
                   default="cross_frontend",
                   help="cross_frontend: the three disagreement statistics; "
                        "all: those plus every predicted metrology quantity")
    p.add_argument("--n-repeats", type=int, default=40,
                   help="random fold assignments used to bound cross-fitting variance")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(OUT_DIR))
    return p.parse_args()


def load_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("L")) > 127


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def build_row(split, name, monitored, metro, masks) -> dict | None:
    row_m = metro[monitored].loc[name]
    others = [m for m in masks if m != monitored]
    if not others:
        return None

    record: dict[str, object] = {"split": split, "name": name}
    for col in row_m.index:
        if not col.startswith("pred_"):
            continue
        try:
            record[col] = float(row_m[col])
        except (TypeError, ValueError):
            continue
    record["disagreement"] = 1.0 - float(
        np.mean([iou(masks[monitored], masks[o]) for o in others])
    )
    for stat, col in (("fg_dev", "pred_foreground_ratio"),
                      ("cc_dev", "pred_component_count")):
        peers = [float(metro[o].loc[name, col]) for o in others if name in metro[o].index]
        record[stat] = abs(float(row_m[col]) - float(np.median(peers))) if peers else np.nan

    # Label: does the monitored frontend's DFM risk call differ from the
    # reference's? The gt_* columns are read here and nowhere else.
    disagree = False
    for q in ("necking_score", "bulging_score"):
        disagree |= (float(row_m[f"pred_{q}"]) > RISK_THRESHOLD) != (
            float(row_m[f"gt_{q}"]) > RISK_THRESHOLD
        )
    record["label"] = bool(disagree)
    return record


def assemble(monitored: str) -> pd.DataFrame:
    folds = json.load(open(FOLDS_JSON))
    rows = []

    for fold_name in folds["folds"]:
        k = int(fold_name.replace("fold", ""))
        metro = {}
        for m in MODELS:
            path = Path(str(CV_METRO).format(k=k, m=m))
            if path.exists():
                metro[m] = pd.read_csv(path, dtype={"name": str}).set_index("name")
        if monitored not in metro or len(metro) < 2:
            continue
        for name in metro[monitored].index:
            if not all(name in metro[m].index for m in metro):
                continue
            masks = {}
            for m in metro:
                mask = load_mask(Path(str(CV_PRED).format(k=k, m=m)) / f"{name}.png")
                if mask is not None:
                    masks[m] = mask
            if monitored in masks and len(masks) >= 2:
                rows.append(build_row("in_dist", name, monitored, metro, masks))

    metro_x = {
        m: pd.read_csv(str(EXTREME_METRO).format(m=m), dtype={"name": str}).set_index("name")
        for m in MODELS
    }
    for name in metro_x[monitored].index:
        masks = {}
        for m in MODELS:
            mask = load_mask(Path(str(EXTREME_PRED).format(m=m)) / f"{name}.png")
            if mask is not None:
                masks[m] = mask
        if monitored in masks and len(masks) >= 2:
            rows.append(build_row("extreme", name, monitored, metro_x, masks))

    return pd.DataFrame([r for r in rows if r is not None])


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 300):
    """Ridge-penalised logistic fit by Newton-Raphson on standardised inputs."""
    Xa = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(Xa.shape[1])
    penalty = l2 * np.eye(Xa.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(iters):
        eta = np.clip(Xa @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-9, None)
        hessian = Xa.T @ (Xa * w[:, None]) + penalty
        step = np.linalg.solve(hessian, Xa.T @ (y - p) - penalty @ beta)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def predict_logistic(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    eta = np.clip(np.column_stack([np.ones(len(X)), X]) @ beta, -30, 30)
    return 1.0 / (1.0 + np.exp(-eta))


def isotonic(x: np.ndarray, y: np.ndarray):
    """Pool-adjacent-violators; returns a callable mapping score -> probability."""
    order = np.argsort(x)
    xs, values = x[order], y[order].astype(float)
    weights = np.ones(len(values))
    i = 0
    while i < len(values) - 1:
        if values[i] <= values[i + 1]:
            i += 1
            continue
        total = weights[i] + weights[i + 1]
        values[i] = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / total
        weights[i] = total
        values = np.delete(values, i + 1)
        weights = np.delete(weights, i + 1)
        xs = np.delete(xs, i + 1)
        i = max(i - 1, 0)
    return lambda q: np.interp(q, xs, values, left=values[0], right=values[-1])


def brier_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> dict:
    """Murphy decomposition: Brier = reliability - resolution + uncertainty."""
    base = float(y.mean())
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    reliability = resolution = ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        if not n:
            continue
        conf, obs = float(p[sel].mean()), float(y[sel].mean())
        reliability += n * (conf - obs) ** 2
        resolution += n * (obs - base) ** 2
        ece += n * abs(conf - obs)
    n_total = len(y)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "reliability": reliability / n_total,
        "resolution": resolution / n_total,
        "uncertainty": base * (1 - base),
        "ece": ece / n_total,
        "base_rate": base,
        "n": n_total,
    }


def reliability_table(p: np.ndarray, y: np.ndarray, split: str, n_bins: int = N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            out.append({
                "split": split,
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "n": int(sel.sum()),
                "mean_predicted": float(p[sel].mean()),
                "observed": float(y[sel].mean()),
            })
    return out


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    r = sps.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = assemble(args.monitored)
    if df.empty:
        raise SystemExit("no records assembled; check that CV metrology outputs exist")
    df.to_csv(out_dir / "records.csv", index=False)

    # Feature set. Repeating the cross-fitting over 40 random fold assignments
    # puts all three candidate sets within noise of one another -- AUROC
    # 0.690 +/- 0.028 for the three cross-frontend statistics alone, 0.692 +/-
    # 0.017 adding morphology and roughness, 0.689 +/- 0.016 for all sixteen --
    # so the thirteen predicted-metrology features buy nothing over the
    # disagreement statistics. We keep the parsimonious set; the comparison was
    # made on in-distribution cross-fitted estimates only.
    candidates = [c for c in df.columns if c not in ("split", "name", "label")]
    available = [c for c in candidates if df[c].isna().mean() < 0.05]
    if args.feature_set == "all":
        feature_cols = available
    else:
        feature_cols = [c for c in ("disagreement", "fg_dev", "cc_dev") if c in available]
    n_before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    dropped = n_before - len(df)

    in_dist = df[df.split == "in_dist"].reset_index(drop=True)
    extreme = df[df.split == "extreme"].reset_index(drop=True)
    if dropped:
        print(f"dropped {dropped} of {n_before} records missing a feature")
    print(f"monitored = {args.monitored}")
    print(f"in-distribution records {len(in_dist)}, Extreme records {len(extreme)}")
    print(f"features {len(feature_cols)} (runtime-only)")
    print(f"base rate: in-dist {in_dist.label.mean():.3f}, Extreme {extreme.label.mean():.3f}\n")

    X_in = in_dist[feature_cols].to_numpy(float)
    y_in = in_dist["label"].to_numpy(bool)
    mu, sd = X_in.mean(0), X_in.std(0)
    sd[sd < 1e-12] = 1.0

    Xs_in = (X_in - mu) / sd

    def cross_fit(seed: int) -> np.ndarray:
        r = np.random.default_rng(seed)
        fold_of = np.zeros(len(in_dist), dtype=int)
        for i, j in enumerate(r.permutation(len(in_dist))):
            fold_of[j] = i % args.n_folds
        out = np.zeros(len(in_dist))
        for k in range(args.n_folds):
            train, test = fold_of != k, fold_of == k
            if not train.any() or not test.any() or len(np.unique(y_in[train])) < 2:
                continue
            beta = fit_logistic(Xs_in[train], y_in[train].astype(float))
            cal = isotonic(predict_logistic(beta, Xs_in[train]), y_in[train])
            out[test] = cal(predict_logistic(beta, Xs_in[test]))
        return out

    # A single fold assignment swings cross-fitted AUROC by ~0.1 at this event
    # count, so report the spread rather than one draw.
    repeats = [cross_fit(int(rng.integers(1 << 31))) for _ in range(args.n_repeats)]
    repeat_auroc = np.array([auroc(p, y_in) for p in repeats])
    repeat_ece = np.array([brier_decomposition(p, y_in)["ece"] for p in repeats])
    p_in = repeats[0]

    # Final model on all in-distribution data, applied unchanged to Extreme.
    beta_full = fit_logistic((X_in - mu) / sd, y_in.astype(float))
    cal_full = isotonic(predict_logistic(beta_full, (X_in - mu) / sd), y_in)
    X_ex = extreme[feature_cols].to_numpy(float)
    y_ex = extreme["label"].to_numpy(bool)
    p_ex = cal_full(predict_logistic(beta_full, (X_ex - mu) / sd))

    summary = {
        "monitored": args.monitored,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "n_folds": args.n_folds,
        "in_dist": {**brier_decomposition(p_in, y_in), "auroc": auroc(p_in, y_in)},
        "extreme": {**brier_decomposition(p_ex, y_ex), "auroc": auroc(p_ex, y_ex)},
        "cross_fit_repeats": {
            "n_repeats": args.n_repeats,
            "auroc_mean": float(repeat_auroc.mean()),
            "auroc_sd": float(repeat_auroc.std()),
            "auroc_min": float(repeat_auroc.min()),
            "auroc_max": float(repeat_auroc.max()),
            "ece_mean": float(repeat_ece.mean()),
        },
    }
    summary["drift"] = {
        "ece_in_dist": summary["in_dist"]["ece"],
        "ece_extreme": summary["extreme"]["ece"],
        "ece_ratio": (
            summary["extreme"]["ece"] / summary["in_dist"]["ece"]
            if summary["in_dist"]["ece"] > 0 else float("inf")
        ),
        "mean_predicted_extreme": float(p_ex.mean()),
        "observed_extreme": float(y_ex.mean()),
    }

    pd.DataFrame(
        reliability_table(p_in, y_in, "in_dist") + reliability_table(p_ex, y_ex, "extreme")
    ).to_csv(out_dir / "reliability.csv", index=False)
    with open(out_dir / "calibration.json", "w") as f:
        json.dump(summary, f, indent=2)

    for split in ("in_dist", "extreme"):
        s = summary[split]
        print(f"=== {split} (n={s['n']}, base rate {s['base_rate']:.3f}) ===")
        print(f"  Brier {s['brier']:.4f} = reliability {s['reliability']:.4f}"
              f" - resolution {s['resolution']:.4f} + uncertainty {s['uncertainty']:.4f}")
        print(f"  ECE   {s['ece']:.4f}    AUROC {s['auroc']:.3f}")
    r = summary["cross_fit_repeats"]
    print(f"\ncross-fitting spread over {r['n_repeats']} fold assignments:"
          f" AUROC {r['auroc_mean']:.3f} +/- {r['auroc_sd']:.3f}"
          f" [{r['auroc_min']:.3f}, {r['auroc_max']:.3f}]")
    d = summary["drift"]
    print(f"calibration drift: ECE {d['ece_in_dist']:.4f} -> {d['ece_extreme']:.4f}"
          f"  ({d['ece_ratio']:.1f}x)")
    print(f"mean predicted risk on Extreme {d['mean_predicted_extreme']:.3f}"
          f" vs observed {d['observed_extreme']:.3f}")
    print(f"\nartifacts -> {out_dir}")


if __name__ == "__main__":
    main()
