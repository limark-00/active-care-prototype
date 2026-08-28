#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/vision/.venv/bin/python"
if ! command -v node >/dev/null 2>&1; then
  BUNDLED_NODE="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin"
  if [ -x "$BUNDLED_NODE/node" ]; then export PATH="$BUNDLED_NODE:$PATH"; fi
fi
if [ ! -x "$PYTHON_BIN" ] || ! command -v pnpm >/dev/null 2>&1; then
  echo '请先按 README.md 安装 Python 虚拟环境和 Node / pnpm。'
  exit 1
fi
cd "$PROJECT_DIR/vision"
"$PYTHON_BIN" download_model.py
VISION_PID=''
WEB_PID=''
cleanup() {
  if [ -n "$VISION_PID" ]; then kill "$VISION_PID" 2>/dev/null || true; fi
  if [ -n "$WEB_PID" ]; then kill "$WEB_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
if curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:8001/health >/dev/null 2>&1; then
  echo '本地视觉服务已在运行。'
else
  "$PYTHON_BIN" -m uvicorn app:app --host 127.0.0.1 --port 8001 --no-access-log &
  VISION_PID=$!
fi
cd "$PROJECT_DIR/web"
if curl --noproxy '*' -fsS --max-time 2 http://localhost:3000/ >/dev/null 2>&1; then
  echo '网页已在运行：http://localhost:3000'
else
  pnpm dev --port 3000 &
  WEB_PID=$!
fi
echo '请打开 http://localhost:3000，点击“开启摄像头与识别”。'
echo '保留此终端；按 Ctrl+C 停止本脚本启动的服务。'
wait
