"""E14: does frontend choice move the process-window boundary?

A fab does not consume CD MAE; it consumes a pass/fail call on whether a
condition sits inside the process window, and then sets margin accordingly.
This script asks what that call looks like when the metrology feeding it comes
from each segmentation frontend instead of from the reference.

Protocol:

1. **Spec limits come from qualified material.** For each metrology quantity
   we take an upper spec limit from the *in-distribution* reference
   distribution (default: its 95th percentile). This is how a fab sets limits
   -- from material already known to be in window -- and it means no Extreme
   sample informs the limit.
2. **Pass/fail per sample.** A sample fails if any quantity exceeds spec. We
   compute this twice: once from the reference masks (truth) and once from
   each frontend's predicted masks.
3. **Boundary along a process-window proxy axis.** We regress fail/pass on
   foreground ratio and report the ratio at which the fitted failure
   probability crosses 0.5. Comparing the boundary implied by a frontend
   against the one implied by the reference gives the shift that frontend
   would introduce.

**The axis is a proxy.** This dataset carries no dose or focus metadata, so
foreground ratio stands in for position within the process window. What is
estimated is therefore the boundary along an observable correlate of window
position, not a dose-focus process window. The direction and relative size of
the shift survive that substitution; a window calibrated in dose-focus units
does not follow from it.

A frontend that places the boundary at a higher foreground ratio than the
reference judges the window narrower than it is, and a fab acting on it would
shrink margin and discard good material. A boundary placed lower means the
window looks wider than it is, and out-of-window material escapes.

Outputs under ``output/revision_v4/e14_pw_boundary/``:

    per_sample.csv   pass/fail per sample under reference and each frontend
    summary.json     spec limits, boundaries, shifts with bootstrap CIs

Usage::

    python scripts/e14_pw_boundary.py --spec-percentile 95
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

IN_DIST = REPO_ROOT / "output/metrology/{m}_test_metrics.csv"
EXTREME = REPO_ROOT / "output/hard_eval/{m}_metrology.csv"
OUT_DIR = REPO_ROOT / "output/revision_v4/e14_pw_boundary"

MODELS = ("unet", "deeplabv3plus", "hrnet", "segformer")
# Quantities carrying an upper spec limit. CD is excluded: it has a target
# rather than a ceiling, so a one-sided limit on it would not mean anything.
SPEC_QUANTITIES = ("lwr_3sigma", "ler_mean_3sigma")
AXIS = "foreground_ratio"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec-percentile", type=float, default=95.0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(OUT_DIR))
    return p.parse_args()


def fit_logistic(x: np.ndarray, y: np.ndarray, iters: int = 200) -> tuple[float, float]:
    """Newton-Raphson logistic fit of y on [1, x]. Returns (intercept, slope)."""
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(2)
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-9, None)
        # Ridge term keeps the Hessian invertible under separation.
        hessian = X.T @ (X * w[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(hessian, X.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def boundary_at_half(x: np.ndarray, fail: np.ndarray) -> float:
    """Axis value where fitted failure probability crosses 0.5.

    Returns nan on a degenerate fit (all pass, all fail, a slope too flat to
    place a crossing, or a crossing outside the observed range) so that callers
    do not read a boundary off noise or off an extrapolation.
    """
    if fail.all() or not fail.any():
        return float("nan")
    intercept, slope = fit_logistic(x, fail.astype(float))
    if abs(slope) < 1e-6:
        return float("nan")
    crossing = -intercept / slope
    if crossing < x.min() or crossing > x.max():
        return float("nan")
    return float(crossing)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Spec limits from the in-distribution reference. One frontend's file is
    # enough because the gt_* columns are identical across frontends.
    in_dist = pd.read_csv(str(IN_DIST).format(m=MODELS[0]))
    spec = {
        q: float(np.percentile(in_dist[f"gt_{q}"].dropna(), args.spec_percentile))
        for q in SPEC_QUANTITIES
    }

    extreme = {m: pd.read_csv(str(EXTREME).format(m=m)) for m in MODELS}
    base = extreme[MODELS[0]]
    axis = base[f"gt_{AXIS}"].to_numpy(float)

    def fails(df: pd.DataFrame, prefix: str) -> np.ndarray:
        out = np.zeros(len(df), dtype=bool)
        for q in SPEC_QUANTITIES:
            out |= df[f"{prefix}_{q}"].to_numpy(float) > spec[q]
        return out

    per_sample = pd.DataFrame({"name": base["name"], AXIS: axis})
    fail_reference = fails(base, "gt")
    per_sample["fail_reference"] = fail_reference
    for m in MODELS:
        per_sample[f"fail_{m}"] = fails(extreme[m], "pred")

    boundary_reference = boundary_at_half(axis, fail_reference)

    summary: dict[str, object] = {
        "spec_percentile": args.spec_percentile,
        "spec_limits": spec,
        "axis": AXIS,
        "axis_is_proxy": (
            "foreground ratio stands in for process-window position; this "
            "dataset carries no dose or focus metadata"
        ),
        "n": int(len(per_sample)),
        "n_fail_reference": int(fail_reference.sum()),
        "boundary_reference": boundary_reference,
        "frontends": {},
    }

    for m in MODELS:
        fail_m = per_sample[f"fail_{m}"].to_numpy(bool)
        boundary_m = boundary_at_half(axis, fail_m)
        shift = boundary_m - boundary_reference

        draws = []
        for _ in range(args.n_boot):
            idx = rng.integers(0, len(axis), len(axis))
            b_ref = boundary_at_half(axis[idx], fail_reference[idx])
            b_m = boundary_at_half(axis[idx], fail_m[idx])
            if np.isfinite(b_ref) and np.isfinite(b_m):
                draws.append(b_m - b_ref)
        lo, hi = (
            np.percentile(draws, [2.5, 97.5]) if len(draws) >= 100 else (float("nan"), float("nan"))
        )

        summary["frontends"][m] = {
            "n_fail": int(fail_m.sum()),
            "boundary": boundary_m,
            "boundary_shift": float(shift),
            "shift_ci_lo": float(lo),
            "shift_ci_hi": float(hi),
            "bootstrap_draws": int(len(draws)),
            # Wafers the frontend would call differently from the reference,
            # split by which way the fab loses.
            "over_reject": int((fail_m & ~fail_reference).sum()),
            "escape": int((~fail_m & fail_reference).sum()),
            "misclassified": int((fail_m != fail_reference).sum()),
            "direction": (
                "narrower than truth (over-shrink, yield loss)"
                if np.isfinite(shift) and shift > 0
                else "wider than truth (escapes)"
                if np.isfinite(shift)
                else "undetermined"
            ),
        }

    per_sample.to_csv(out_dir / "per_sample.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"spec limits from in-distribution reference, P{args.spec_percentile:g}:")
    for q, v in spec.items():
        print(f"  {q:<20} <= {v:.4f}")
    print(f"\nExtreme: n = {summary['n']}, reference calls {summary['n_fail_reference']} out of spec")
    print(f"reference boundary (foreground ratio, PROXY axis) = {boundary_reference:.4f}\n")

    header = (
        f"{'frontend':<15}{'n_fail':>8}{'boundary':>10}{'shift':>9}"
        f"{'95% CI':>20}{'over-rej':>10}{'escape':>8}"
    )
    print(header)
    for m, s in summary["frontends"].items():
        ci = f"[{s['shift_ci_lo']:+.3f}, {s['shift_ci_hi']:+.3f}]"
        print(
            f"{m:<15}{s['n_fail']:>8}{s['boundary']:>10.4f}{s['boundary_shift']:>+9.4f}"
            f"{ci:>20}{s['over_reject']:>10}{s['escape']:>8}"
        )
    print(f"\nartifacts -> {out_dir}")


if __name__ == "__main__":
    main()
