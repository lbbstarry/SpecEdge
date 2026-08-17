"""E20: benchmark the guard against published reference-free risk signals.

Section II cites the uncertainty and reference-free quality-estimation
literature, but the guard has so far only been compared against two statistics
we defined ourselves. This script closes that gap by scoring the established
image-level signals on exactly the same images, labels and groupings:

    msp        one minus the mean max-softmax probability, the confidence
               baseline of Hendrycks and Gimpel
    entropy    mean per-pixel predictive entropy
    mcdropout  mean per-pixel variance of the foreground probability over T
               stochastic forward passes with dropout left on (Gal and
               Ghahramani). Only available for frontends carrying an active
               dropout layer, which here is DeepLabV3+ and SegFormer.
    ensemble   mean per-pixel variance of the foreground probability across
               the co-deployed frontends, the deep-ensemble signal of
               Lakshminarayanan et al. This differs from the guard's score,
               which compares thresholded masks by IoU rather than
               probabilities.

Every signal is aggregated to one number per image in two ways: over all
pixels, and over a band around the predicted boundary, since segmentation
uncertainty concentrates there and a whole-image mean can be swamped by easy
background. Both are reported; neither is selected on the evaluation split.

The failure label, the splits and the bootstrap procedure are taken unchanged
from the guard evaluation, so the numbers are directly comparable to those in
Section VIII-G.

Outputs under ``output/revision_v4/e20_uncertainty_baselines/``:

    scores.csv    per-image score for every signal
    summary.json  AUROC with bootstrap CI per signal, per setting

Usage::

    python scripts/e20_uncertainty_baselines.py --mc-passes 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats as sps

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.baselines._cfg import load_config  # noqa: E402
from specedge.baselines import build_model  # noqa: E402

OUT_DIR = REPO_ROOT / "output/revision_v4/e20_uncertainty_baselines"
DISAGREEMENT = REPO_ROOT / "output/revision_v4/e4_disagreement.csv"
NOISE_FLOOR = REPO_ROOT / "output/revision_v4/e1b_noise_floor.json"

MODELS = ("unet", "deeplabv3plus", "hrnet", "segformer")
SPLITS = {
    "standard": REPO_ROOT / "dataset/litho/images/test",
    "extreme": REPO_ROOT / "dataset/litho_hard/images/hard",
}
CKPT = REPO_ROOT / "output/baselines/{m}/best.pth"
IMAGE_SIZE = 512
BOUNDARY_BAND = 5  # pixels each side of the predicted boundary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mc-passes", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=str(OUT_DIR))
    return p.parse_args()


def load_image(path: Path, device: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return t.to(device)


def foreground_prob(logits: torch.Tensor) -> torch.Tensor:
    """Foreground probability map, for one- or two-channel heads."""
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)[:, 0]
    return F.softmax(logits, dim=1)[:, 1]


def enable_dropout(model: torch.nn.Module) -> int:
    """Put dropout layers back in train mode; returns how many were activated."""
    n = 0
    for m in model.modules():
        if "dropout" in type(m).__name__.lower() and getattr(m, "p", 0) > 0:
            m.train()
            n += 1
    return n


def boundary_mask(prob: np.ndarray, band: int = BOUNDARY_BAND) -> np.ndarray:
    """Pixels within `band` of the predicted decision boundary."""
    hard = prob > 0.5
    return ndi.binary_dilation(hard, iterations=band) & ~ndi.binary_erosion(
        hard, iterations=band
    )


def aggregate(pixel_score: np.ndarray, band: np.ndarray) -> tuple[float, float]:
    whole = float(pixel_score.mean())
    edge = float(pixel_score[band].mean()) if band.any() else float("nan")
    return whole, edge


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg) or not np.isfinite(score).all():
        return float("nan")
    r = sps.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cluster_bootstrap(
    df: pd.DataFrame, col: str, label: np.ndarray, rng: np.random.Generator, n_boot: int
) -> np.ndarray:
    names = df["name"].to_numpy()
    uniq = np.unique(names)
    idx_by = {n: np.flatnonzero(names == n) for n in uniq}
    vals = df[col].to_numpy(float)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[n] for n in pick])
        a = auroc(vals[idx], label[idx])
        if np.isfinite(a):
            draws.append(a)
    return np.asarray(draws)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows: dict[tuple[str, str, str], dict] = {}
    prob_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for model in MODELS:
        cfg = load_config(str(REPO_ROOT / f"scripts/baselines/configs/{model}.yaml"))
        net = build_model(cfg["model"], num_classes=cfg["data"]["num_classes"]).to(args.device)
        state = torch.load(str(CKPT).format(m=model), map_location=args.device)
        net.load_state_dict(state["model"] if "model" in state else state)
        net.eval()
        n_dropout = sum(
            1 for m in net.modules()
            if "dropout" in type(m).__name__.lower() and getattr(m, "p", 0) > 0
        )

        for split, img_dir in SPLITS.items():
            for path in sorted(img_dir.iterdir()):
                if path.suffix.lower() not in (".png", ".bmp", ".jpg"):
                    continue
                name = path.stem
                x = load_image(path, args.device)
                with torch.no_grad():
                    prob = foreground_prob(net(x))[0].cpu().numpy()

                prob_cache.setdefault((split, name), {})[model] = prob
                band = boundary_mask(prob)

                p = np.clip(prob, 1e-6, 1 - 1e-6)
                msp_pix = 1.0 - np.maximum(p, 1 - p)
                ent_pix = -(p * np.log(p) + (1 - p) * np.log(1 - p))
                msp_w, msp_e = aggregate(msp_pix, band)
                ent_w, ent_e = aggregate(ent_pix, band)

                rec = {
                    "split": split, "model": model, "name": name,
                    "msp_whole": msp_w, "msp_edge": msp_e,
                    "entropy_whole": ent_w, "entropy_edge": ent_e,
                }

                if n_dropout:
                    enable_dropout(net)
                    with torch.no_grad():
                        stack = np.stack([
                            foreground_prob(net(x))[0].cpu().numpy()
                            for _ in range(args.mc_passes)
                        ])
                    net.eval()
                    rec["mcdropout_whole"], rec["mcdropout_edge"] = aggregate(
                        stack.var(axis=0), band
                    )
                else:
                    rec["mcdropout_whole"] = rec["mcdropout_edge"] = np.nan

                rows[(split, model, name)] = rec
        print(f"  scored {model} (active dropout layers: {n_dropout})")

    # Deep-ensemble variance needs every frontend's probability map for an image.
    for (split, name), probs in prob_cache.items():
        if len(probs) < 2:
            continue
        var = np.stack([probs[m] for m in sorted(probs)]).var(axis=0)
        for model in probs:
            key = (split, model, name)
            if key in rows:
                rows[key]["ensemble_whole"], rows[key]["ensemble_edge"] = aggregate(
                    var, boundary_mask(probs[model])
                )

    scores = pd.DataFrame(rows.values())
    scores.to_csv(out_dir / "scores.csv", index=False)

    dis = pd.read_csv(DISAGREEMENT, dtype={"name": str}).dropna(subset=["abs_err_cd_mean"])
    scores["name"] = scores["name"].astype(str)
    df = dis.merge(scores, on=["split", "model", "name"], how="inner")
    with open(NOISE_FLOOR) as f:
        tau = float(json.load(f)["extreme"]["systematic_1px"]["cd_mean"]["mean"])

    signals = [
        ("disagreement (ours)", "disagreement"),
        ("fg deviation", "pred_fg_dev"),
        ("cc deviation", "pred_cc_dev"),
        ("MSP", "msp_whole"),
        ("MSP, boundary band", "msp_edge"),
        ("entropy", "entropy_whole"),
        ("entropy, boundary band", "entropy_edge"),
        ("MC-dropout var", "mcdropout_whole"),
        ("MC-dropout var, boundary", "mcdropout_edge"),
        ("ensemble var", "ensemble_whole"),
        ("ensemble var, boundary", "ensemble_edge"),
    ]
    settings = {
        "extreme_segformer": (df.split == "extreme") & (df.model == "segformer"),
        "extreme_all_models": df.split == "extreme",
        "pooled_all": df.split.notna(),
    }

    summary: dict[str, object] = {"tau_px": tau, "n_boot": args.n_boot, "settings": {}}
    print(f"\njoined records: {len(df)}   tau = {tau:.3f} px\n")
    for setting, sel in settings.items():
        block = df[sel]
        label = block["abs_err_cd_mean"].to_numpy(float) > tau
        entry: dict[str, object] = {
            "n": int(len(block)), "n_fail": int(label.sum()), "signals": {},
        }
        print(f"=== {setting}  (n={len(block)}, failures={int(label.sum())}) ===")
        print(f"{'signal':<28}{'AUROC':>8}{'95% CI':>20}{'n':>7}")
        for pretty, col in signals:
            sub = block.dropna(subset=[col])
            if sub.empty:
                continue
            lab = sub["abs_err_cd_mean"].to_numpy(float) > tau
            point = auroc(sub[col].to_numpy(float), lab)
            draws = cluster_bootstrap(sub, col, lab, rng, args.n_boot)
            lo, hi = (np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan))
            entry["signals"][pretty] = {
                "auroc": point, "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(sub)),
            }
            print(f"{pretty:<28}{point:>8.3f}   [{lo:>6.3f}, {hi:>6.3f}]{len(sub):>7}")
        summary["settings"][setting] = entry
        print()

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
