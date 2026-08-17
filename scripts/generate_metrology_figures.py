"""Generate paper-oriented metrology diagnostic figures.

This script consumes existing baseline predictions and metrology CSV files:

    output/baselines/{model}/preds/
    output/metrology/{model}_test_metrics.csv

and writes figures under output/metrology/figures by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from specedge.metrology import binarize_mask, connected_components, estimate_dominant_orientation


@dataclass
class RawProfile:
    label: int
    centroid: float
    axis: np.ndarray
    width: np.ndarray
    low_edge: np.ndarray
    high_edge: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="dataset/litho")
    p.add_argument("--baseline-root", default="output/baselines")
    p.add_argument("--metrology-root", default="output/metrology")
    p.add_argument("--output-dir", default="output/metrology/figures")
    p.add_argument("--models", nargs="+", default=["unet", "deeplabv3plus", "hrnet", "segformer"])
    p.add_argument("--display-model", default="segformer")
    p.add_argument("--cases", nargs="+", default=None)
    p.add_argument("--split", default="test")
    return p.parse_args()


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def resize_like(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    h, w = shape
    return np.asarray(Image.fromarray(mask).resize((w, h), Image.NEAREST))


def resize_rgb(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == shape:
        return image
    h, w = shape
    return np.asarray(Image.fromarray(image).resize((w, h), Image.BILINEAR))


def detrend(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return values - values.mean() if values.size else values
    x = np.arange(values.size, dtype=np.float64)
    coeff = np.polyfit(x, values, deg=1)
    return values - (coeff[0] * x + coeff[1])


def psd_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = detrend(values)
    if residual.size < 4:
        return np.asarray([]), np.asarray([])
    power = np.abs(np.fft.rfft(residual)) ** 2
    if power.size <= 1:
        return np.asarray([]), np.asarray([])
    power = power[1:]
    freq = np.linspace(0, 1, num=power.size, endpoint=True)
    total = power.sum()
    if total > 0:
        power = power / total
    return freq, power


def component_profiles(mask: np.ndarray, min_area: int = 64) -> list[RawProfile]:
    binary = binarize_mask(mask)
    orientation = estimate_dominant_orientation(binary, min_area=min_area)
    if orientation not in {"horizontal", "vertical"}:
        return []
    labels, keep = connected_components(binary, min_area=min_area)
    profiles: list[RawProfile] = []
    for label in keep:
        component = labels == label
        ys, xs = np.nonzero(component)
        if ys.size == 0:
            continue
        y0, x0, y1, x1 = int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1
        crop = component[y0:y1, x0:x1]
        axis: list[float] = []
        width: list[float] = []
        low_edge: list[float] = []
        high_edge: list[float] = []
        if orientation == "horizontal":
            for xi in range(crop.shape[1]):
                loc = np.nonzero(crop[:, xi])[0]
                if loc.size == 0:
                    continue
                low = float(loc.min() + y0)
                high = float(loc.max() + y0)
                axis.append(float(xi + x0))
                low_edge.append(low)
                high_edge.append(high)
                width.append(high - low + 1.0)
            centroid = float(ys.mean())
        else:
            for yi in range(crop.shape[0]):
                loc = np.nonzero(crop[yi, :])[0]
                if loc.size == 0:
                    continue
                low = float(loc.min() + x0)
                high = float(loc.max() + x0)
                axis.append(float(yi + y0))
                low_edge.append(low)
                high_edge.append(high)
                width.append(high - low + 1.0)
            centroid = float(xs.mean())
        if len(width) >= 8:
            profiles.append(
                RawProfile(
                    label=label,
                    centroid=centroid,
                    axis=np.asarray(axis),
                    width=np.asarray(width),
                    low_edge=np.asarray(low_edge),
                    high_edge=np.asarray(high_edge),
                )
            )
    return profiles


def choose_profile_pair(gt: np.ndarray, pred: np.ndarray) -> tuple[RawProfile | None, RawProfile | None]:
    gt_profiles = component_profiles(gt)
    pred_profiles = component_profiles(pred)
    if not gt_profiles or not pred_profiles:
        return None, None
    gt_profile = max(gt_profiles, key=lambda p: p.axis.size)
    pred_profile = min(pred_profiles, key=lambda p: abs(p.centroid - gt_profile.centroid))
    return gt_profile, pred_profile


def read_metric_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def identity_limits(values_a: list[float], values_b: list[float]) -> tuple[float, float]:
    vals = np.asarray(values_a + values_b, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(vals.min())
    hi = float(vals.max())
    pad = max((hi - lo) * 0.08, 1e-3)
    return lo - pad, hi + pad


def plot_scatter_all_models(models: list[str], metrology_root: Path, output_dir: Path) -> Path:
    metric_specs = [
        ("cd_mean", "CD mean (px)"),
        ("lwr_3sigma", "LWR 3sigma (px)"),
        ("ler_mean_3sigma", "LER mean 3sigma (px)"),
        ("edge_psd_hf_ratio_1d", "Edge PSD HF ratio"),
    ]
    colors = {
        "unet": "#4C78A8",
        "deeplabv3plus": "#F58518",
        "hrnet": "#54A24B",
        "segformer": "#B279A2",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=160)
    axes = axes.ravel()
    for ax, (metric, label) in zip(axes, metric_specs):
        all_gt: list[float] = []
        all_pred: list[float] = []
        for model in models:
            rows = read_metric_rows(metrology_root / f"{model}_test_metrics.csv")
            gt = [as_float(r, f"gt_{metric}") for r in rows]
            pred = [as_float(r, f"pred_{metric}") for r in rows]
            ok = np.isfinite(gt) & np.isfinite(pred)
            gt_ok = np.asarray(gt)[ok]
            pred_ok = np.asarray(pred)[ok]
            all_gt.extend(gt_ok.tolist())
            all_pred.extend(pred_ok.tolist())
            ax.scatter(
                gt_ok,
                pred_ok,
                s=18,
                alpha=0.62,
                label=model,
                color=colors.get(model),
                edgecolors="none",
            )
        lo, hi = identity_limits(all_gt, all_pred)
        ax.plot([lo, hi], [lo, hi], color="#333333", linewidth=1.0, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(label)
        ax.set_xlabel("GT-derived")
        ax.set_ylabel("Pred-derived")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Pred-derived vs GT-derived Metrology", fontsize=14)
    fig.tight_layout()
    out = output_dir / "scatter_metrology_all_models.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_mae_summary(models: list[str], metrology_root: Path, output_dir: Path) -> Path:
    summary_path = metrology_root / "summary_metrology.csv"
    rows = read_metric_rows(summary_path)
    metrics = [
        ("cd_mean_mae", "CD mean MAE"),
        ("lwr_3sigma_mae", "LWR 3sigma MAE"),
        ("ler_mean_3sigma_mae", "LER mean 3sigma MAE"),
        ("edge_psd_hf_ratio_1d_mae", "PSD HF ratio MAE"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), dpi=160)
    for ax, (key, title) in zip(axes, metrics):
        vals = [as_float(next(r for r in rows if r["model"] == model), key) for model in models]
        ax.bar(models, vals, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(models)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Metrology Error Summary")
    fig.tight_layout()
    out = output_dir / "mae_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def case_paths(data_root: Path, baseline_root: Path, model: str, split: str, name: str) -> dict[str, Path]:
    return {
        "image": data_root / "images" / split / f"{name}.png",
        "gt": data_root / "masks" / split / f"{name}.png",
        "pred": baseline_root / model / "preds" / "masks" / f"{name}.png",
        "overlay": baseline_root / model / "preds" / "overlays" / f"{name}.png",
        "error": baseline_root / model / "preds" / "errors" / f"{name}.png",
    }


def plot_case_diagnostic(
    data_root: Path,
    baseline_root: Path,
    metrology_root: Path,
    output_dir: Path,
    model: str,
    split: str,
    name: str,
) -> Path | None:
    paths = case_paths(data_root, baseline_root, model, split, name)
    if not all(path.exists() for path in paths.values()):
        return None

    pred = binarize_mask(load_mask(paths["pred"]))
    gt = binarize_mask(resize_like(load_mask(paths["gt"]), pred.shape))
    image = resize_rgb(load_rgb(paths["image"]), pred.shape)
    overlay = load_rgb(paths["overlay"])
    error = load_rgb(paths["error"])
    gt_profile, pred_profile = choose_profile_pair(gt, pred)

    fig = plt.figure(figsize=(13, 9), dpi=160)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_overlay = fig.add_subplot(gs[0, 1])
    ax_error = fig.add_subplot(gs[0, 2])
    for ax, arr, title in [
        (ax_img, image, "SEM image"),
        (ax_overlay, overlay, "Boundary overlay"),
        (ax_error, error, "Error map"),
    ]:
        ax.imshow(arr)
        ax.set_title(title)
        ax.axis("off")

    ax_width = fig.add_subplot(gs[1, 0])
    ax_edge = fig.add_subplot(gs[1, 1])
    ax_psd = fig.add_subplot(gs[1, 2])
    if gt_profile is not None and pred_profile is not None:
        ax_width.plot(gt_profile.axis, gt_profile.width, label="GT", linewidth=1.5, color="#2CA02C")
        ax_width.plot(pred_profile.axis, pred_profile.width, label="Pred", linewidth=1.2, color="#D62728")
        ax_width.set_title("Representative width profile")
        ax_width.set_xlabel("Scan position (px)")
        ax_width.set_ylabel("Width (px)")
        ax_width.grid(True, alpha=0.25)
        ax_width.legend(frameon=False, fontsize=8)

        gt_edge = detrend((gt_profile.low_edge + gt_profile.high_edge) / 2.0)
        pred_edge = detrend((pred_profile.low_edge + pred_profile.high_edge) / 2.0)
        ax_edge.plot(gt_profile.axis[: gt_edge.size], gt_edge, label="GT", linewidth=1.5, color="#2CA02C")
        ax_edge.plot(pred_profile.axis[: pred_edge.size], pred_edge, label="Pred", linewidth=1.2, color="#D62728")
        ax_edge.set_title("Detrended edge center residual")
        ax_edge.set_xlabel("Scan position (px)")
        ax_edge.set_ylabel("Residual (px)")
        ax_edge.grid(True, alpha=0.25)

        gt_freq, gt_psd = psd_curve(gt_edge)
        pred_freq, pred_psd = psd_curve(pred_edge)
        if gt_freq.size:
            ax_psd.semilogy(gt_freq, gt_psd + 1e-12, label="GT", linewidth=1.5, color="#2CA02C")
        if pred_freq.size:
            ax_psd.semilogy(pred_freq, pred_psd + 1e-12, label="Pred", linewidth=1.2, color="#D62728")
        ax_psd.set_title("Normalized 1D edge PSD")
        ax_psd.set_xlabel("Normalized frequency")
        ax_psd.set_ylabel("Power")
        ax_psd.grid(True, alpha=0.25)
        ax_psd.legend(frameon=False, fontsize=8)
    else:
        for ax in (ax_width, ax_edge, ax_psd):
            ax.text(0.5, 0.5, "No measurable profile", ha="center", va="center")
            ax.axis("off")

    metric_rows = read_metric_rows(metrology_root / f"{model}_test_metrics.csv")
    row = next((r for r in metric_rows if r["name"] == name), None)
    subtitle = ""
    if row:
        subtitle = (
            f"CD err={as_float(row, 'abs_err_cd_mean'):.3f}px, "
            f"LWR err={as_float(row, 'abs_err_lwr_3sigma'):.3f}px, "
            f"LER err={as_float(row, 'abs_err_ler_mean_3sigma'):.3f}px"
        )
    fig.suptitle(f"{model} case {name}  {subtitle}", fontsize=14)
    fig.tight_layout()
    out = output_dir / "case_diagnostics" / f"{model}_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def default_cases(data_root: Path) -> list[str]:
    subset_path = data_root / "subsets.json"
    if subset_path.exists():
        payload = json.load(subset_path.open())
        names = list(payload.get("hard", []))
        names.extend(payload.get("needs_visual_review", []))
        return list(dict.fromkeys(names))
    return ["00000051", "00000511", "00000750"]


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    baseline_root = Path(args.baseline_root)
    metrology_root = Path(args.metrology_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    outputs.append(plot_scatter_all_models(args.models, metrology_root, output_dir))
    outputs.append(plot_mae_summary(args.models, metrology_root, output_dir))

    cases = args.cases or default_cases(data_root)
    for name in cases:
        out = plot_case_diagnostic(
            data_root=data_root,
            baseline_root=baseline_root,
            metrology_root=metrology_root,
            output_dir=output_dir,
            model=args.display_model,
            split=args.split,
            name=name,
        )
        if out is not None:
            outputs.append(out)

    print("Generated figures:")
    for out in outputs:
        print(f"  {out}")


if __name__ == "__main__":
    main()

