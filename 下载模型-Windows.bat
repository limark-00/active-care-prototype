@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0下载模型-Windows.ps1"
if errorlevel 1 (
  echo.
  echo 模型下载失败，请查看上方错误。
  pause
  exit /b 1
)
echo.
echo 模型下载和校验完成。
pause
