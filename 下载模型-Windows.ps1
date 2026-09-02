$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonBin = Join-Path $ProjectDir "vision\.venv\Scripts\python.exe"

if (-not (Test-Path $PythonBin)) {
    Write-Error "找不到 vision\.venv。请先按 README.md 的 Windows 步骤安装 Python 3.12 环境与依赖。"
    exit 1
}

& $PythonBin (Join-Path $ProjectDir "scripts\download_models.py")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "模型准备完成。可以按 README.md 启动后端和网页。" -ForegroundColor Green
