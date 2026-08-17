"""Summarize hard-set baseline and metrology evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_MODELS = ["unet", "deeplabv3plus", "hrnet", "segformer"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", default="output/hard_eval")
    p.add_argument("--standard-gt", default="output/metrology/gt_test_metrics.csv")
    p.add_argument("--hard-gt", default="output/metrology/hard_new_gt_metrics.csv")
    p.add_argument("--output-dir", default="output/hard_eval")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def f(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(rows: list[dict[str, str]], key: str) -> float:
    vals = [f(row.get(key)) for row in rows]
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def load_json(path: Path) -> dict:
    with path.open() as file:
        return json.load(file)


def gt_distribution_rows(standard_path: Path, hard_path: Path) -> list[dict[str, object]]:
    specs = [
        ("cd_mean", "CD mean"),
        ("lwr_3sigma", "LWR 3sigma"),
        ("ler_mean_3sigma", "LER mean 3sigma"),
        ("edge_psd_hf_ratio_1d", "Edge PSD HF ratio"),
        ("component_count", "Component count"),
    ]
    standard = read_csv(standard_path)
    hard = read_csv(hard_path)
    rows: list[dict[str, object]] = []
    for key, label in specs:
        rows.append(
            {
                "metric": label,
                "standard_test_mean": f"{mean(standard, key):.6g}",
                "hard_set_mean": f"{mean(hard, key):.6g}",
                "hard_over_standard": f"{mean(hard, key) / max(mean(standard, key), 1e-12):.6g}",
            }
        )
    return rows


def segmentation_summary(eval_root: Path, models: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in models:
        payload = load_json(eval_root / f"{model}_eval.json")
        s = payload["summary"]
        rows.append(
            {
                "model": model,
                "num_samples": s.get("num_samples"),
                "iou": f"{f(s.get('iou')):.6g}",
                "dice": f"{f(s.get('dice')):.6g}",
                "boundary_f1": f"{f(s.get('boundary_f1')):.6g}",
                "hausdorff95": f"{f(s.get('hausdorff95')):.6g}",
                "psd_pred_gt_ratio": f"{f(s.get('edge_psd_hf_ratio_pred')) / max(f(s.get('edge_psd_hf_ratio_gt')), 1e-12):.6g}",
                "inference_s_per_image": f"{f(s.get('inference_s_per_image')):.6g}",
            }
        )
    return rows


def metrology_summary(eval_root: Path, models: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in models:
        payload = load_json(eval_root / f"{model}_metrology_summary.json")
        s = payload["summary"]
        rows.append(
            {
                "model": model,
                "num_samples": s.get("num_samples"),
                "cd_mean_mae": f"{f(s.get('cd_mean_mae')):.6g}",
                "cd_mean_pearson": f"{f(s.get('cd_mean_pearson')):.6g}",
                "lwr_3sigma_mae": f"{f(s.get('lwr_3sigma_mae')):.6g}",
                "lwr_3sigma_pearson": f"{f(s.get('lwr_3sigma_pearson')):.6g}",
                "ler_mean_3sigma_mae": f"{f(s.get('ler_mean_3sigma_mae')):.6g}",
                "ler_mean_3sigma_pearson": f"{f(s.get('ler_mean_3sigma_pearson')):.6g}",
                "edge_psd_hf_ratio_1d_mae": f"{f(s.get('edge_psd_hf_ratio_1d_mae')):.6g}",
                "edge_psd_hf_ratio_1d_pearson": f"{f(s.get('edge_psd_hf_ratio_1d_pearson')):.6g}",
                "component_count_mae": f"{f(s.get('component_count_mae')):.6g}",
                "open_candidate_count": s.get("open_candidate_count"),
                "necking_candidate_count": s.get("necking_candidate_count"),
                "bulging_candidate_count": s.get("bulging_candidate_count"),
            }
        )
    return rows


def worst_case_rows(eval_root: Path, models: list[str], top_k: int = 8) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in models:
        metric_rows = read_csv(eval_root / f"{model}_metrology.csv")
        metric_rows = sorted(metric_rows, key=lambda r: f(r.get("abs_err_cd_mean"), -1.0), reverse=True)
        for rank, row in enumerate(metric_rows[:top_k], start=1):
            rows.append(
                {
                    "model": model,
                    "rank": rank,
                    "name": row.get("name"),
                    "abs_err_cd_mean": row.get("abs_err_cd_mean"),
                    "abs_err_lwr_3sigma": row.get("abs_err_lwr_3sigma"),
                    "abs_err_ler_mean_3sigma": row.get("abs_err_ler_mean_3sigma"),
                    "abs_err_component_count": row.get("abs_err_component_count"),
                    "pred_status": row.get("pred_status"),
                    "gt_status": row.get("gt_status"),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    out = Path(args.output_dir)

    outputs = {
        "gt_distribution_compare.csv": gt_distribution_rows(Path(args.standard_gt), Path(args.hard_gt)),
        "summary_segmentation.csv": segmentation_summary(eval_root, args.models),
        "summary_metrology.csv": metrology_summary(eval_root, args.models),
        "worst_cases_by_cd_error.csv": worst_case_rows(eval_root, args.models),
    }
    for filename, rows in outputs.items():
        write_csv(rows, out / filename)
        print(f"wrote {len(rows)} rows -> {out / filename}")


if __name__ == "__main__":
    main()
