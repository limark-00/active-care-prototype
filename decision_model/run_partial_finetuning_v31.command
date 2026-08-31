#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/vision/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo '找不到 vision/.venv，请先按项目README安装Python环境。'
  exit 1
fi
cd "$ROOT"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
"$PYTHON" -m pip install -r decision_model/training/requirements.txt
"$PYTHON" -m decision_model.generate_dataset_v31
"$PYTHON" -m decision_model.training.train_partial_macbert \
  --data-dir "$ROOT/decision_model/data/v3_1" \
  --device auto \
  --aux-risk-head \
  --selection-metric safety_weighted \
  --ordinal-loss-weight 0.25 \
  --underprediction-loss-weight 0.15
printf '\nV3.1部分微调结束。日志位于：%s\n' "$ROOT/decision_model/outputs"
