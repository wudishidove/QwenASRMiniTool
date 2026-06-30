@echo off
chcp 65001 > nul
setlocal

REM ============================================================
REM   Qwen3 ASR GPU Launcher  (Official long-audio chunking)
REM   This wrapper enables qwen-asr's official low-energy splitter
REM   inside app-gpu.py, then delegates to start-gpu.bat.
REM ============================================================

set "QWEN_GPU_SEGMENT_MODE=official"
echo  [MODE] Official low-energy chunking enabled.
echo         Default chunk target: 5 minutes (configurable in the UI).
echo.

call "%~dp0start-gpu.bat"
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
