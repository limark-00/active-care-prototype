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
"$PYTHON" -m decision_model.generate_dataset_v3
"$PYTHON" -m decision_model.training.train_partial_macbert --device auto
printf '\n部分微调结束。日志位于：%s\n' "$ROOT/decision_model/outputs"
