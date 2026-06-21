"""version.py — 應用程式版本與更新來源設定（單一事實來源）

此檔同時被 app.py / app-gpu.py / setting.py / updater.py 引用。
打包時務必以 --add-data 一併納入 EXE，讓自動更新能正確比對版本。

版本規則：
    語意化版本 MAJOR.MINOR.PATCH。
    每次發佈新編譯版時，先把 __version__ 往上加（例如 1.0.6 → 1.0.7），
    再到 GitHub 建立同名 tag 的 Release，並上傳整包 ZIP 資產。
"""
from __future__ import annotations

# 本次編譯版本（dist2）。1.0.9：新增 Whisper/Breeze-ASR-26 推理核心
# (CrispASR / Vulkan，-nf -bo1 達 ~35x，繁中/台語特化)、qwen3 ForcedAligner
# GGUF 字級時間軸對齊、核心切換 Vulkan context 防當機修正、字幕分行全引擎統一
# (subtitle_lines，標點/空白邊界切 + MAX_CHARS 保護 + 孤兒行合併)。
# 最新已發佈 Release 為 1.0.8。
__version__ = "1.0.9"

# 自動更新來源：GitHub repo（owner/name）
GITHUB_REPO = "dseditor/QwenASRMiniTool"

# GitHub Releases API（latest 端點，回傳最新「非預發佈」版本）
GITHUB_API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

# 發行頁（供「前往下載頁」按鈕使用）
GITHUB_RELEASES_PAGE = f"https://github.com/{GITHUB_REPO}/releases/latest"
