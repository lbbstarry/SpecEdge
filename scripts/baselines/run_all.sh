#!/usr/bin/env bash
# 批量训练并评估必跑 baseline.
# 用法: bash scripts/baselines/run_all.sh [DATA_ROOT] [OUT_ROOT]

set -euo pipefail

DATA_ROOT="${1:-dataset/litho}"
OUT_ROOT="${2:-output/baselines}"
CONFIG_DIR="scripts/baselines/configs"

# 第一阶段必跑模型. 跑通后再追加 pointrend / mask2former.
MODELS=(unet deeplabv3plus hrnet segformer)

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.csv"
echo "model,iou,dice,boundary_f1,hausdorff95,edge_psd_hf_ratio_pred,edge_psd_hf_ratio_gt,inference_s_per_image" > "$SUMMARY"

for m in "${MODELS[@]}"; do
    cfg="$CONFIG_DIR/${m}.yaml"
    out="$OUT_ROOT/$m"
    echo ">>> train $m"
    python scripts/baselines/train_baseline.py --config "$cfg" --data-root "$DATA_ROOT" --output "$out"

    echo ">>> eval $m on test"
    python scripts/baselines/eval_baseline.py --config "$cfg" --ckpt "$out/best.pth" \
        --data-root "$DATA_ROOT" --split test --output "$out/eval_test.json"

    python - <<PY >> "$SUMMARY"
import json
s = json.load(open("$out/eval_test.json"))["summary"]
keys = ["iou","dice","boundary_f1","hausdorff95","edge_psd_hf_ratio_pred","edge_psd_hf_ratio_gt","inference_s_per_image"]
print("$m," + ",".join(f"{s.get(k, float('nan')):.6f}" for k in keys))
PY
done

echo ">>> all done. summary -> $SUMMARY"
