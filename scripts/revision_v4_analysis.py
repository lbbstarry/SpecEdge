"""Revision-v4 analysis bundle (paper revision plan v4, E1a/E1b/E3/E4/E7/E8/E5b/E10).

All analyses run from cached masks and per-sample CSVs; no GPU required.

  E1a  CNN-consensus alternative reference (majority vote of U-Net/DeepLabV3+/HRNet)
  E1b  Reference perturbation noise floor (systematic +-1px, stochastic boundary noise)
  E3   Failure onset redo: Extreme-only segmented regression + bootstrap CI + logistic
  E4   GT-free cross-frontend disagreement guard (Spearman + AUROC)
  E7   IoU-bin conditional spread of metrology errors
  E8   Layout-bbox coverage confound (r_bbox)
  E5b  Training-support analysis (train fg-ratio histogram)
  E10  Numeric reconciliation of the two PSD quantities

Outputs: output/revision_v4/  (CSVs, JSONs, PNG figures, consensus masks)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from specedge.metrology import evaluate_metrology_pair, mask_metrology

OUT = REPO / "output/revision_v4"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["unet", "deeplabv3plus", "hrnet", "segformer"]
SPLITS = {
    "standard": {
        "metro": REPO / "output/metrology/{m}_test_metrics.csv",
        "iou": REPO / "output/baselines/{m}/eval_test.json",
        "pred": REPO / "output/baselines/{m}/preds/masks",
        "gt": REPO / "dataset/litho/masks/test",
    },
    "extreme": {
        "metro": REPO / "output/hard_eval/{m}_metrology.csv",
        "iou": REPO / "output/hard_eval/{m}_eval.json",
        "pred": REPO / "output/hard_eval/{m}/preds/masks",
        "gt": REPO / "dataset/litho_hard/masks/hard",
    },
}
LAYOUT_DIR = REPO / "dataset/litho_hard/layout_manual/hard"
TRAIN_MASK_DIR = REPO / "dataset/litho/masks/train"
RNG = np.random.default_rng(42)
SUMMARY: dict[str, object] = {}


def load_bin(path: Path) -> np.ndarray:
    m = np.asarray(Image.open(path).convert("L"))
    return (m > 127).astype(np.uint8)


def resize_like(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(np.uint8)
    h, w = shape
    out = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST))
    return (out > 127).astype(np.uint8)


def metro_df(split: str, model: str) -> pd.DataFrame:
    return pd.read_csv(str(SPLITS[split]["metro"]).format(m=model), dtype={"name": str})


def iou_df(split: str, model: str) -> pd.DataFrame:
    with open(str(SPLITS[split]["iou"]).format(m=model)) as f:
        d = json.load(f)
    return pd.DataFrame(
        [{"name": str(s["name"]), "iou": s["iou"],
          "psd2d_pred": s.get("edge_psd_hf_ratio_pred"), "psd2d_gt": s.get("edge_psd_hf_ratio_gt")}
         for s in d["per_sample"]]
    )


def names_of(split: str) -> list[str]:
    return sorted(p.stem for p in Path(SPLITS[split]["gt"]).glob("*.png"))


def iou_of(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _f(value: object) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------- E1a consensus
def e1a_consensus() -> None:
    print("== E1a: CNN-consensus alternative reference ==")
    res = {}
    for split in SPLITS:
        gt_dir = Path(SPLITS[split]["gt"])
        cons_dir = OUT / f"consensus_ref/{split}"
        cons_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for name in names_of(split):
            gt = load_bin(gt_dir / f"{name}.png")
            preds = {}
            for m in MODELS:
                p = load_bin(Path(str(SPLITS[split]["pred"]).format(m=m)) / f"{name}.png")
                preds[m] = resize_like(p, gt.shape)
            cons = ((preds["unet"] + preds["deeplabv3plus"] + preds["hrnet"]) >= 2).astype(np.uint8)
            Image.fromarray(cons * 255).save(cons_dir / f"{name}.png")
            sf_vs_cons = evaluate_metrology_pair(preds["segformer"] * 255, cons * 255)
            cons_vs_ref = evaluate_metrology_pair(cons * 255, gt * 255)
            rows.append({
                "name": name,
                "cons_vs_ref_iou": iou_of(cons, gt),
                "cons_vs_ref_cd_mae": _f(cons_vs_ref.get("abs_err_cd_mean")),
                "cons_vs_ref_lwr_mae": _f(cons_vs_ref.get("abs_err_lwr_3sigma")),
                "cons_vs_ref_ler_mae": _f(cons_vs_ref.get("abs_err_ler_mean_3sigma")),
                "sf_vs_cons_cd_mae": _f(sf_vs_cons.get("abs_err_cd_mean")),
                "sf_vs_cons_lwr_mae": _f(sf_vs_cons.get("abs_err_lwr_3sigma")),
                "sf_vs_cons_ler_mae": _f(sf_vs_cons.get("abs_err_ler_mean_3sigma")),
            })
        df = pd.DataFrame(rows)
        ref = metro_df(split, "segformer")[
            ["name", "abs_err_cd_mean", "abs_err_lwr_3sigma", "abs_err_ler_mean_3sigma", "gt_foreground_ratio"]
        ].rename(columns={
            "abs_err_cd_mean": "sf_vs_lithoref_cd_mae",
            "abs_err_lwr_3sigma": "sf_vs_lithoref_lwr_mae",
            "abs_err_ler_mean_3sigma": "sf_vs_lithoref_ler_mae",
        })
        df = df.merge(ref, on="name")
        df.to_csv(OUT / f"e1a_consensus_{split}.csv", index=False)
        both = df.dropna(subset=["sf_vs_cons_cd_mae", "sf_vs_lithoref_cd_mae"])
        rho = stats.spearmanr(both["sf_vs_cons_cd_mae"], both["sf_vs_lithoref_cd_mae"])
        k = 10
        w_cons = set(both.nlargest(k, "sf_vs_cons_cd_mae")["name"])
        w_ref = set(both.nlargest(k, "sf_vs_lithoref_cd_mae")["name"])
        res[split] = {
            "n": len(df),
            "consensus_vs_lithoref_iou_mean": float(df["cons_vs_ref_iou"].mean()),
            "reference_disagreement_cd_mae": float(np.nanmean(df["cons_vs_ref_cd_mae"])),
            "reference_disagreement_lwr_mae": float(np.nanmean(df["cons_vs_ref_lwr_mae"])),
            "reference_disagreement_ler_mae": float(np.nanmean(df["cons_vs_ref_ler_mae"])),
            "segformer_cd_mae_vs_consensus": float(both["sf_vs_cons_cd_mae"].mean()),
            "segformer_cd_mae_vs_lithoref": float(both["sf_vs_lithoref_cd_mae"].mean()),
            "segformer_lwr_mae_vs_consensus": float(both["sf_vs_cons_lwr_mae"].mean()),
            "segformer_ler_mae_vs_consensus": float(both["sf_vs_cons_ler_mae"].mean()),
            "spearman_err_cons_vs_err_lithoref": float(rho.statistic),
            "worst10_overlap": len(w_cons & w_ref),
            "worst10_consensus": sorted(w_cons),
            "worst10_lithoref": sorted(w_ref),
        }
        print(f"  {split}: SegFormer CD MAE vs consensus {res[split]['segformer_cd_mae_vs_consensus']:.3f} px "
              f"(vs LithoSeg-ref {res[split]['segformer_cd_mae_vs_lithoref']:.3f} px), "
              f"err Spearman {res[split]['spearman_err_cons_vs_err_lithoref']:.3f}, "
              f"worst10 overlap {res[split]['worst10_overlap']}/10")
    json.dump(res, open(OUT / "e1a_summary.json", "w"), indent=2)
    SUMMARY["e1a"] = res


# ---------------------------------------------------------------- E1b noise floor
def perturb_metrics(gt: np.ndarray) -> dict[str, dict[str, float]]:
    keys = ["cd_mean", "lwr_3sigma", "ler_mean_3sigma", "edge_psd_hf_ratio_1d"]
    kernel = np.ones((3, 3), np.uint8)
    base = mask_metrology(gt * 255)
    if base.get("status") != "ok":
        return {}
    out: dict[str, dict[str, float]] = {}
    dil = cv2.dilate(gt, kernel, iterations=1)
    ero = cv2.erode(gt, kernel, iterations=1)
    sys_d = {}
    for k in keys:
        ds = []
        for pert in (dil, ero):
            m = mask_metrology(pert * 255)
            if m.get("status") == "ok" and np.isfinite(_f(m.get(k))) and np.isfinite(_f(base.get(k))):
                ds.append(abs(_f(m[k]) - _f(base[k])))
        if ds:
            sys_d[k] = float(np.mean(ds))
    out["systematic_1px"] = sys_d
    band = (dil > 0) & (ero == 0)
    sto_d: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(3):
        flip = band & (RNG.random(gt.shape) < 0.25)
        noisy = np.where(flip, 1 - gt, gt).astype(np.uint8)
        m = mask_metrology(noisy * 255)
        if m.get("status") != "ok":
            continue
        for k in keys:
            if np.isfinite(_f(m.get(k))) and np.isfinite(_f(base.get(k))):
                sto_d[k].append(abs(_f(m[k]) - _f(base[k])))
    out["stochastic_band"] = {k: float(np.mean(v)) for k, v in sto_d.items() if v}
    return out


def e1b_noise_floor() -> None:
    print("== E1b: reference perturbation noise floor ==")
    res = {}
    for split in SPLITS:
        gt_dir = Path(SPLITS[split]["gt"])
        acc: dict[str, dict[str, list[float]]] = {}
        for name in names_of(split):
            pm = perturb_metrics(load_bin(gt_dir / f"{name}.png"))
            for mode, d in pm.items():
                for k, v in d.items():
                    acc.setdefault(mode, {}).setdefault(k, []).append(v)
        res[split] = {mode: {k: {"mean": float(np.mean(v)), "median": float(np.median(v))}
                             for k, v in d.items()} for mode, d in acc.items()}
        cd_sys = res[split].get("systematic_1px", {}).get("cd_mean", {}).get("mean", float("nan"))
        cd_sto = res[split].get("stochastic_band", {}).get("cd_mean", {}).get("mean", float("nan"))
        print(f"  {split}: sigma_ref CD systematic +-1px = {cd_sys:.3f} px; stochastic = {cd_sto:.4f} px")
    json.dump(res, open(OUT / "e1b_noise_floor.json", "w"), indent=2)
    SUMMARY["e1b"] = res


# ---------------------------------------------------------------- E3 onset redo
def two_piece_fit(x: np.ndarray, y: np.ndarray, min_side: int = 5):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    best = None
    ux = np.unique(xs)
    for i in range(len(ux) - 1):
        b = (ux[i] + ux[i + 1]) / 2
        left, right = xs <= b, xs > b
        if left.sum() < min_side or right.sum() < min_side:
            continue
        sse = 0.0
        for sel in (left, right):
            c = np.polyfit(xs[sel], ys[sel], 1)
            sse += float(((np.polyval(c, xs[sel]) - ys[sel]) ** 2).sum())
        if best is None or sse < best[1]:
            best = (b, sse)
    return best


def neg_loglik(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    z = X @ beta
    return float(np.sum(np.logaddexp(0, z)) - np.sum(y * z) + 1e-6 * np.sum(beta**2))


def fit_logistic(X: np.ndarray, y: np.ndarray):
    r = minimize(neg_loglik, np.zeros(X.shape[1]), args=(X, y), method="Nelder-Mead",
                 options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
    return r.x, -neg_loglik(r.x, X, y)


def e3_onset(rbbox: pd.DataFrame, tau: float) -> None:
    print(f"== E3: onset redo (Extreme-only, tau={tau:.2f} px) ==")
    sf = metro_df("extreme", "segformer").dropna(subset=["abs_err_cd_mean", "gt_foreground_ratio"])
    x = sf["gt_foreground_ratio"].to_numpy(float)
    y = np.log10(sf["abs_err_cd_mean"].to_numpy(float) + 1e-3)
    bp, sse2 = two_piece_fit(x, y)
    c1 = np.polyfit(x, y, 1)
    sse1 = float(((np.polyval(c1, x) - y) ** 2).sum())
    n = len(x)
    fstat = ((sse1 - sse2) / 3) / (sse2 / (n - 5))
    pval = float(stats.f.sf(fstat, 3, n - 5))
    boots = []
    for _ in range(1000):
        idx = RNG.integers(0, n, n)
        r = two_piece_fit(x[idx], y[idx])
        if r is not None:
            boots.append(r[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    below, above = x <= bp, x > bp
    err = sf["abs_err_cd_mean"].to_numpy(float)
    std_sf = metro_df("standard", "segformer").dropna(subset=["abs_err_cd_mean", "gt_foreground_ratio"])
    std_overlap = std_sf[std_sf["gt_foreground_ratio"] <= max(hi, bp)]
    fail = err > tau
    rb = sf.merge(rbbox, on="name", how="left")
    X = np.column_stack([np.ones(n), stats.zscore(x), stats.zscore(rb["r_bbox"].to_numpy(float))])
    yb = fail.astype(float)
    full, ll_full = fit_logistic(X, yb)
    _, ll_nofg = fit_logistic(X[:, [0, 2]], yb)
    _, ll_norb = fit_logistic(X[:, [0, 1]], yb)
    res = {
        "n_extreme": n, "tau_px": tau,
        "breakpoint": float(bp), "breakpoint_ci95": [float(lo), float(hi)],
        "f_test_p_two_piece_vs_linear": pval,
        "cd_mae_mean_below_bp": float(err[below].mean()), "n_below": int(below.sum()),
        "cd_mae_mean_above_bp": float(err[above].mean()), "n_above": int(above.sum()),
        "n_fail": int(fail.sum()),
        "fg_min_extreme": float(x.min()), "fg_min_standard": float(std_sf["gt_foreground_ratio"].min()),
        "standard_overlap_n": int(len(std_overlap)),
        "standard_overlap_cd_mae_max": float(std_overlap["abs_err_cd_mean"].max()),
        "standard_overlap_cd_mae_median": float(std_overlap["abs_err_cd_mean"].median()),
        "logistic": {
            "coef_fg_z": float(full[1]), "coef_rbbox_z": float(full[2]),
            "lr_p_fg": float(stats.chi2.sf(2 * (ll_full - ll_nofg), 1)),
            "lr_p_rbbox": float(stats.chi2.sf(2 * (ll_full - ll_norb), 1)),
        },
    }
    json.dump(res, open(OUT / "e3_onset.json", "w"), indent=2)
    SUMMARY["e3"] = res
    print(f"  breakpoint {bp:.3f}  CI95 [{lo:.3f}, {hi:.3f}]  F-test p={pval:.2e}")
    print(f"  CD MAE below/above bp: {res['cd_mae_mean_below_bp']:.2f} / {res['cd_mae_mean_above_bp']:.2f} px;"
          f" standard overlap n={res['standard_overlap_n']} max err {res['standard_overlap_cd_mae_max']:.3f}")
    print(f"  logistic: fg z-coef {full[1]:.2f} (p={res['logistic']['lr_p_fg']:.3g}), "
          f"r_bbox z-coef {full[2]:.2f} (p={res['logistic']['lr_p_rbbox']:.3g})")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(std_sf["gt_foreground_ratio"], std_sf["abs_err_cd_mean"] + 1e-3, s=18, alpha=0.6,
               label="standard test (in-window)", color="#4878d0")
    ax.scatter(x, err + 1e-3, s=22, alpha=0.8, label="Extreme (out-of-window)", color="#d65f5f")
    ax.axvspan(lo, hi, color="orange", alpha=0.25, label=f"onset bp 95% CI [{lo:.2f},{hi:.2f}]")
    ax.axhline(tau, ls="--", c="gray", lw=1, label=f"tau = {tau:.1f} px")
    ax.set_yscale("log"); ax.set_xlabel("GT foreground ratio"); ax.set_ylabel("SegFormer per-sample CD MAE (px)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_e3_onset.png", dpi=160); plt.close(fig)


# ---------------------------------------------------------------- E4 disagreement guard
def e4_guard(tau: float) -> None:
    print(f"== E4: GT-free cross-frontend disagreement guard (tau={tau:.2f} px) ==")
    rows = []
    for split in SPLITS:
        metros = {m: metro_df(split, m).set_index("name") for m in MODELS}
        for name in names_of(split):
            mm = {m: load_bin(Path(str(SPLITS[split]["pred"]).format(m=m)) / f"{name}.png") for m in MODELS}
            shape = min((v.shape for v in mm.values()))
            mm = {k: resize_like(v, shape) for k, v in mm.items()}
            for m in MODELS:
                others = [o for o in MODELS if o != m]
                d = 1.0 - float(np.mean([iou_of(mm[m], mm[o]) for o in others]))
                row = {"split": split, "model": m, "name": name, "disagreement": d}
                if name in metros[m].index:
                    r = metros[m].loc[name]
                    row["abs_err_cd_mean"] = _f(r["abs_err_cd_mean"])
                    row["pred_fg_dev"] = abs(_f(r["pred_foreground_ratio"]) -
                                             np.median([_f(metros[o].loc[name, "pred_foreground_ratio"]) for o in others]))
                    row["pred_cc_dev"] = abs(_f(r["pred_component_count"]) -
                                             np.median([_f(metros[o].loc[name, "pred_component_count"]) for o in others]))
                rows.append(row)
    df = pd.DataFrame(rows).dropna(subset=["abs_err_cd_mean"])
    df.to_csv(OUT / "e4_disagreement.csv", index=False)
    res = {}
    for scope, sel in [("extreme_segformer", (df.split == "extreme") & (df.model == "segformer")),
                       ("extreme_all_models", df.split == "extreme"),
                       ("pooled_all", df.split.notna())]:
        sub = df[sel]
        lab = sub["abs_err_cd_mean"].to_numpy(float) > tau
        res[scope] = {
            "n": len(sub), "n_fail": int(lab.sum()),
            "spearman_d_vs_err": float(stats.spearmanr(sub["disagreement"], sub["abs_err_cd_mean"]).statistic),
            "auroc_disagreement": auroc(sub["disagreement"].to_numpy(float), lab),
            "auroc_fg_dev": auroc(sub["pred_fg_dev"].to_numpy(float), lab),
            "auroc_cc_dev": auroc(sub["pred_cc_dev"].to_numpy(float), lab),
        }
        print(f"  {scope}: n={len(sub)} fail={int(lab.sum())} AUROC d={res[scope]['auroc_disagreement']:.3f} "
              f"(fg_dev {res[scope]['auroc_fg_dev']:.3f}, cc_dev {res[scope]['auroc_cc_dev']:.3f}), "
              f"Spearman {res[scope]['spearman_d_vs_err']:.3f}")
    json.dump(res, open(OUT / "e4_summary.json", "w"), indent=2)
    SUMMARY["e4"] = res
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for split, c in [("standard", "#4878d0"), ("extreme", "#d65f5f")]:
        s = df[(df.split == split) & (df.model == "segformer")]
        axes[0].scatter(s["disagreement"], s["abs_err_cd_mean"] + 1e-3, s=20, alpha=0.7, c=c, label=split)
    axes[0].set_yscale("log"); axes[0].set_xlabel("disagreement vs other frontends (1 - mean IoU)")
    axes[0].set_ylabel("SegFormer CD MAE (px)"); axes[0].legend(fontsize=8)
    sub = df[df.split == "extreme"]
    lab = sub["abs_err_cd_mean"].to_numpy(float) > tau
    sc = sub["disagreement"].to_numpy(float)
    ths = np.unique(sc)
    tpr = [(sc[lab] >= t).mean() for t in ths]
    fpr = [(sc[~lab] >= t).mean() for t in ths]
    axes[1].plot(fpr, tpr, "-o", ms=2.5, c="#d65f5f",
                 label=f"Extreme all models, AUROC={res['extreme_all_models']['auroc_disagreement']:.3f}")
    axes[1].plot([0, 1], [0, 1], "--", c="gray", lw=1)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_e4_guard.png", dpi=160); plt.close(fig)


# ---------------------------------------------------------------- E7 IoU-bin spread
def e7_iou_bins() -> None:
    print("== E7: IoU-bin conditional spread (standard test) ==")
    frames = []
    for m in MODELS:
        d = iou_df("standard", m).merge(metro_df("standard", m), on="name")
        d["model"] = m
        frames.append(d)
    df = pd.concat(frames)
    bins = [0.97, 0.98, 0.985, 0.99, 0.995, 1.0]
    rows = []
    for i in range(len(bins) - 1):
        sub = df[(df.iou >= bins[i]) & (df.iou < bins[i + 1])]
        if len(sub) < 5:
            continue
        for metric in ["abs_err_cd_mean", "abs_err_lwr_3sigma", "abs_err_ler_mean_3sigma"]:
            v = sub[metric].dropna().to_numpy(float)
            p5, p50, p95 = np.percentile(v, [5, 50, 95])
            rows.append({"iou_bin": f"[{bins[i]}, {bins[i+1]})", "n": len(v), "metric": metric,
                         "p5": p5, "p50": p50, "p95": p95,
                         "spread_p95_over_p5": p95 / max(p5, 1e-6)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "e7_iou_bin_spread.csv", index=False)
    cd = out[out.metric == "abs_err_cd_mean"]
    SUMMARY["e7"] = {"max_cd_spread_factor": float(cd["spread_p95_over_p5"].max()),
                     "bins": cd.to_dict("records")}
    print(out.to_string(index=False))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, c in zip(MODELS, ["#4878d0", "#ee854a", "#6acc64", "#d65f5f"]):
        s = df[df.model == m]
        ax.scatter(s["iou"], s["abs_err_cd_mean"] + 1e-3, s=16, alpha=0.65, c=c, label=m)
    ax.set_yscale("log"); ax.set_xlabel("per-sample IoU"); ax.set_ylabel("per-sample CD MAE (px)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_e7_iou_vs_cd.png", dpi=160); plt.close(fig)


# ---------------------------------------------------------------- E8 r_bbox
def e8_rbbox() -> pd.DataFrame:
    print("== E8: layout-bbox coverage confound ==")
    rows = []
    ex_dir = OUT / "e8_examples"
    ex_dir.mkdir(exist_ok=True)
    for i, name in enumerate(names_of("extreme")):
        img = np.asarray(Image.open(LAYOUT_DIR / f"{name}.png").convert("L"))
        th, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # The layout sketches are drawn dark-on-light throughout, so strokes are
        # always below the Otsu threshold. Deciding polarity from the image mean
        # instead selects the background on the 47 of 65 sketches whose drawn
        # area covers more than half the canvas, saturating r_bbox at 1.0.
        stroke = img < th
        stroke = cv2.dilate(stroke.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
        ncc, _, cc_stats, _ = cv2.connectedComponentsWithStats(stroke, 8)
        canvas = np.zeros_like(stroke)
        for j in range(1, ncc):
            x0, y0, w, h, area = cc_stats[j]
            if area < 100:
                continue
            canvas[y0:y0 + h, x0:x0 + w] = 1
        rows.append({"name": name, "r_bbox": float(canvas.mean())})
        if i < 3:
            Image.fromarray((stroke * 127 + canvas * 128).astype(np.uint8)).save(ex_dir / f"{name}_strokes_bbox.png")
    rb = pd.DataFrame(rows)
    sf = metro_df("extreme", "segformer")[["name", "abs_err_cd_mean", "gt_foreground_ratio"]]
    df = rb.merge(sf, on="name").dropna()
    r_fg = stats.spearmanr(df["r_bbox"], df["gt_foreground_ratio"])
    r_err = stats.spearmanr(df["r_bbox"], df["abs_err_cd_mean"])
    rk = df[["r_bbox", "gt_foreground_ratio", "abs_err_cd_mean"]].rank()
    res_b = rk["r_bbox"] - np.polyval(np.polyfit(rk["gt_foreground_ratio"], rk["r_bbox"], 1), rk["gt_foreground_ratio"])
    res_e = rk["abs_err_cd_mean"] - np.polyval(np.polyfit(rk["gt_foreground_ratio"], rk["abs_err_cd_mean"], 1), rk["gt_foreground_ratio"])
    partial = stats.pearsonr(res_b, res_e)
    df.to_csv(OUT / "e8_rbbox.csv", index=False)
    SUMMARY["e8"] = {
        "spearman_rbbox_vs_fg": float(r_fg.statistic), "p_fg": float(r_fg.pvalue),
        "spearman_rbbox_vs_err": float(r_err.statistic), "p_err": float(r_err.pvalue),
        "partial_rbbox_vs_err_given_fg": float(partial.statistic), "p_partial": float(partial.pvalue),
    }
    print(f"  Spearman(r_bbox, fg)={r_fg.statistic:.3f} (p={r_fg.pvalue:.3g}); "
          f"(r_bbox, err)={r_err.statistic:.3f} (p={r_err.pvalue:.3g}); "
          f"partial|fg={partial.statistic:.3f} (p={partial.pvalue:.3g})")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sc = ax.scatter(df["r_bbox"], df["abs_err_cd_mean"] + 1e-3, c=df["gt_foreground_ratio"], cmap="viridis", s=26)
    fig.colorbar(sc, label="GT foreground ratio")
    ax.set_yscale("log"); ax.set_xlabel("layout-bbox coverage r_bbox"); ax.set_ylabel("SegFormer CD MAE (px)")
    fig.tight_layout(); fig.savefig(OUT / "fig_e8_rbbox.png", dpi=160); plt.close(fig)
    return rb


# ---------------------------------------------------------------- E5b train support
def e5b_train_support() -> None:
    print("== E5b: training-support analysis ==")
    fgs = np.asarray([float(load_bin(p).mean()) for p in sorted(TRAIN_MASK_DIR.glob("*.png"))])
    ex = metro_df("extreme", "segformer")["gt_foreground_ratio"].dropna().to_numpy(float)
    st = metro_df("standard", "segformer")["gt_foreground_ratio"].dropna().to_numpy(float)
    SUMMARY["e5b"] = {
        "train_n": len(fgs), "train_fg_min": float(fgs.min()),
        "train_fg_p1": float(np.percentile(fgs, 1)), "train_fg_p5": float(np.percentile(fgs, 5)),
        "extreme_below_train_min": int((ex < fgs.min()).sum()),
        "standard_below_train_min": int((st < fgs.min()).sum()),
    }
    print(f"  train fg min={fgs.min():.3f} p1={np.percentile(fgs, 1):.3f}; "
          f"Extreme below train min: {SUMMARY['e5b']['extreme_below_train_min']}; "
          f"standard below train min: {SUMMARY['e5b']['standard_below_train_min']}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fgs, bins=40, alpha=0.6, label=f"train (n={len(fgs)})", color="#82c6e2", density=True)
    ax.hist(st, bins=20, alpha=0.5, label="standard test", color="#4878d0", density=True)
    ax.hist(ex, bins=20, alpha=0.5, label="Extreme", color="#d65f5f", density=True)
    ax.axvline(fgs.min(), ls="--", c="k", lw=1, label=f"train min = {fgs.min():.3f}")
    ax.set_xlabel("GT foreground ratio"); ax.set_ylabel("density"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_e5b_train_support.png", dpi=160); plt.close(fig)
    json.dump(SUMMARY["e5b"], open(OUT / "e5b_train_support.json", "w"), indent=2)


# ---------------------------------------------------------------- E10 numeric PSD
def e10_psd() -> None:
    print("== E10: PSD definition reconciliation (numeric) ==")
    res = {}
    for split in SPLITS:
        per_model = {}
        for m in MODELS:
            d2 = iou_df(split, m)
            d1 = metro_df(split, m)
            per_model[m] = {
                "psd2d_pred_over_gt": float((d2["psd2d_pred"] / d2["psd2d_gt"]).mean()),
                "psd1d_pred_mean": float(d1["pred_edge_psd_hf_ratio_1d"].mean()),
                "psd1d_gt_mean": float(d1["gt_edge_psd_hf_ratio_1d"].mean()),
                "psd1d_pred_over_gt": float((d1["pred_edge_psd_hf_ratio_1d"] / d1["gt_edge_psd_hf_ratio_1d"]).mean()),
            }
        res[split] = per_model
    json.dump(res, open(OUT / "e10_psd_reconciliation.json", "w"), indent=2)
    SUMMARY["e10"] = res
    for split, pm in res.items():
        sf = pm["segformer"]
        print(f"  {split}/segformer: 2D boundary-map ratio {sf['psd2d_pred_over_gt']:.3f}, "
              f"1D edge-residual ratio {sf['psd1d_pred_over_gt']:.3f}")


def main() -> None:
    e1a_consensus()
    e1b_noise_floor()
    tau = SUMMARY["e1b"]["extreme"]["systematic_1px"]["cd_mean"]["mean"]
    SUMMARY["tau_px"] = tau
    rbbox = e8_rbbox()
    e3_onset(rbbox, tau)
    e4_guard(tau)
    e7_iou_bins()
    e5b_train_support()
    e10_psd()
    json.dump(SUMMARY, open(OUT / "summary_revision_v4.json", "w"), indent=2, default=str)
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
