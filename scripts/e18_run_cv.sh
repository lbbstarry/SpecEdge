#!/usr/bin/env bash
# E18: train + evaluate the four frontends across every cross-validation fold.
#
# Wraps scripts/baselines/run_all.sh once per fold produced by
# scripts/e18_make_cv_folds.py, so each of the 588 in-distribution images is
# scored exactly once as held-out data (the published result uses a single
# 60-image test split).
#
# A fold whose summary.csv already lists all four models is skipped, so the
# run can be resumed after an interruption without retraining.
#
# Usage: bash scripts/e18_run_cv.sh [CV_ROOT] [OUT_ROOT]

set -euo pipefail

CV_ROOT="${1:-dataset/litho_cv}"
OUT_ROOT="${2:-output/cv}"
N_MODELS=4

if [ ! -f "$CV_ROOT/folds.json" ]; then
    echo "error: $CV_ROOT/folds.json not found; run scripts/e18_make_cv_folds.py first" >&2
    exit 1
fi

K=$(python -c "import json;print(json.load(open('$CV_ROOT/folds.json'))['k'])")
echo "=== E18 cross-validation: $K folds x $N_MODELS frontends ==="
mkdir -p "$OUT_ROOT"

for ((k = 0; k < K; k++)); do
    fold_data="$CV_ROOT/fold$k"
    fold_out="$OUT_ROOT/fold$k"
    summary="$fold_out/summary.csv"

    # Header line plus one row per model means this fold already finished.
    if [ -f "$summary" ] && [ "$(wc -l < "$summary")" -ge $((N_MODELS + 1)) ]; then
        echo ">>> fold$k already complete, skipping"
        continue
    fi

    echo ">>> fold$k  ($fold_data -> $fold_out)"
    bash scripts/baselines/run_all.sh "$fold_data" "$fold_out"
done

echo "=== E18 done. per-fold summaries under $OUT_ROOT/fold*/summary.csv ==="
