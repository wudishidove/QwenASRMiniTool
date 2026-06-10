@echo off
REM =======================================================
REM  make_release_zip.bat
REM  Package dist2\QwenASR into a versioned ZIP for a
REM  GitHub Release. The in-app updater downloads this ZIP
REM  and overlay-copies it over the user's install, so the
REM  archive must contain ONLY program files - NOT the large
REM  model / venv folders (those are downloaded at runtime).
REM
REM  OUTPUT: QwenASR_<version>.zip  (repo root)
REM =======================================================
setlocal

set "SRC=F:\AIStudio\QwenASR"
set "APPDIR=%SRC%\dist2\QwenASR"
set "STAGE=%TEMP%\QwenASR_pkg"

IF NOT EXIST "%APPDIR%\QwenASR.exe" (
    echo [ERROR] %APPDIR%\QwenASR.exe not found.
    echo         Run build.bat first to produce dist2\QwenASR\.
    pause & exit /b 1
)

REM ---- Resolve version from version.py -------------------
IF EXIST "%SRC%\build_venv\Scripts\python.exe" (
    set "PYTHON=%SRC%\build_venv\Scripts\python.exe"
) ELSE (
    set "PYTHON=%SRC%\venv\Scripts\python.exe"
)
REM cd to SRC so "import version" resolves. The venv python path has no
REM spaces, so it needs no quoting inside for/f (avoids nested-quote issues).
cd /d "%SRC%"
for /f "delims=" %%v in ('%PYTHON% -c "import version;print(version.__version__)"') do set "VER=%%v"
IF "%VER%"=="" set "VER=dev"
echo  Version: %VER%

set "OUTZIP=%SRC%\QwenASR_%VER%.zip"

REM ---- Stage program files (exclude big runtime data) ----
echo  Staging program files (excluding models / venvs)...
IF EXIST "%STAGE%" rmdir /S /Q "%STAGE%"
mkdir "%STAGE%\QwenASR"

REM robocopy mirrors the app dir but skips large / per-user folders.
REM /XD = exclude these directories anywhere in the tree.
robocopy "%APPDIR%" "%STAGE%\QwenASR" /E ^
    /XD "GPUModel" "ov_models" "venv-gpu" "venv" "subtitles" "__pycache__" ^
    /XF "settings.json" "*.log" ^
    /NFL /NDL /NJH /NJS /NP >nul

REM ---- Compress to ZIP -----------------------------------
echo  Compressing to %OUTZIP% ...
IF EXIST "%OUTZIP%" del "%OUTZIP%"
powershell -NoProfile -Command "Compress-Archive -Path '%STAGE%\QwenASR' -DestinationPath '%OUTZIP%' -CompressionLevel Optimal -Force"
IF ERRORLEVEL 1 (
    echo [ERROR] Compression failed.
    rmdir /S /Q "%STAGE%" 2>nul
    pause & exit /b 1
)

rmdir /S /Q "%STAGE%" 2>nul

echo.
echo ===================================================
echo  Release ZIP ready:
echo    %OUTZIP%
echo.
echo  Next steps:
echo    1. Create a GitHub Release with tag  %VER%
echo       on  dseditor/QwenASRMiniTool
echo    2. Upload  QwenASR_%VER%.zip  as a release asset
echo    3. Deployed apps (older version) will detect it via
echo       Settings - Check Update and self-update.
echo ===================================================
pause
