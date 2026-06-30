"""version.py — 應用程式版本與更新來源設定（單一事實來源）

此檔同時被 app.py / app-gpu.py / setting.py / updater.py 引用。
打包時務必以 --add-data 一併納入 EXE，讓自動更新能正確比對版本。

版本規則：
    語意化版本 MAJOR.MINOR.PATCH。
    每次發佈新編譯版時，先把 __version__ 往上加（例如 1.0.6 → 1.0.7），
    再到 GitHub 建立同名 tag 的 Release，並上傳整包 ZIP 資產。
"""
from __future__ import annotations

# 本次編譯版本（dist2）。1.0.10：GPU CUDA workflow V4 紀錄。
# 新增 start-gpu-official.bat 官方低能量長音訊切片入口，並讓 start-gpu.bat
# 同時支援發行包 cudagpu\ 與原始碼根目錄 layout；GPUModel/ 會自動掃描可用
# Qwen3-ASR 模型並提供下拉切換，預設偏好 pkm-ft-1.7b-v2。
# GPU 版新增 OpenCC 繁化主開關、空白斷句開關與模型相依預設；ForcedAligner
# 字幕時間軸改以 raw_text 字元對位，避免 tokenizer 丟符號時掉字或標點錯位。
# 前一版為 1.0.9。
__version__ = "1.0.10"

# 自動更新來源：GitHub repo（owner/name）
GITHUB_REPO = "dseditor/QwenASRMiniTool"

# GitHub Releases API（latest 端點，回傳最新「非預發佈」版本）
GITHUB_API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

# 發行頁（供「前往下載頁」按鈕使用）
GITHUB_RELEASES_PAGE = f"https://github.com/{GITHUB_REPO}/releases/latest"
