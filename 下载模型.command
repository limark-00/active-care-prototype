#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/vision/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo '找不到 vision/.venv。请先按 README.md 安装 Python 3.12 环境与依赖。'
  exit 1
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/download_models.py"
echo '模型准备完成。可以双击“启动本地伴护.command”。'
