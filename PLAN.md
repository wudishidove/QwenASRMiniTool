# V4 版本紀錄：GPU 微調模型與官方切片

## Version
- App version: `v1.0.10`
- Internal milestone: `~V4`
- Scope: CUDA GPU source workflow (`app-gpu.py` / `start-gpu*.bat`)

## Summary
這版把 GPU 工作流整理成可切換模型、可保留微調模型原文、可走官方長音訊切片的 V4 紀錄版。重點是讓 `pkm-ft-1.7b-v2` 這類微調模型可以直接在 Source GPU 版中使用，同時保留原本 VAD 路徑，另以 `start-gpu-official.bat` 啟用 qwen-asr 官方 low-energy splitter。

## Key Changes
- 新增 `start-gpu-official.bat`：
  - 啟動前設定 `QWEN_GPU_SEGMENT_MODE=official`。
  - 委派既有 `start-gpu.bat` 啟動同一個 `app-gpu.py`。
  - 官方切片預設目標長度為 5 分鐘，可在 UI 調整。

- 更新 `start-gpu.bat`：
  - 同時支援 release build 的 `cudagpu\` layout 與 source clone 根目錄 layout。
  - 找不到 `app-gpu.py` 時顯示兩種預期路徑，方便排查。

- 更新 `app-gpu.py` 的 GPU 檔案轉字幕流程：
  - `GPUASREngine.process_file()` 在 official mode 下改用 `qwen_asr.inference.utils.split_audio_into_chunks()`。
  - diarization 開啟時，會先依 speaker 段落切分，再對各 speaker 長段套用官方切片。
  - SRT 時間軸仍優先使用現有 ForcedAligner，失敗或未啟用時 fallback 到比例估算。
  - 即時錄音仍沿用既有 VAD 流程。

- 新增 GPU 模型選擇：
  - 掃描 `GPUModel/` 底下含 `config.json` 且 `model_type=qwen3_asr` 的模型。
  - 模型分頁提供 GPU 模型下拉選單。
  - 預設偏好 `pkm-ft-1.7b-v2`，否則沿用原始 `Qwen3-ASR-1.7B` 或第一個可用模型。
  - 切換模型時會保存 `gpu_asr_model` 並重新載入模型。

- 新增 GPU 版輸出控制：
  - 設定分頁加入 OpenCC 繁化主開關，微調模型可關閉以保留繁體專名與模型原文。
  - 加入「空白也斷句」開關，支援早期無標點微調模型。
  - 依模型名稱自動推定 OpenCC / 空白斷句預設值。
  - 新增設定 key：`opencc_enabled`、`break_on_space`、`official_chunk_minutes`。

- 修正 ForcedAligner 字幕切行時間軸：
  - 改以 `raw_text` 字元為準建立時間對位。
  - tokenizer 丟棄符號（例如 `%`）時，不再因 word-list 重建文字而掉字或讓標點掛錯詞。
  - fallback 與 aligner 路徑都會合併過短的孤兒行。

## Test Notes
- 靜態檢查：`python -m py_compile app-gpu.py model_tab.py setting.py version.py`
- 建議手動煙霧測試：
  - 執行 `start-gpu.bat`，確認仍走原本 VAD 模式。
  - 執行 `start-gpu-official.bat`，確認 UI 顯示官方切片分鐘欄位。
  - 使用短音檔與長音檔各跑一次 SRT 輸出。
  - 切換 `pkm-ft-1.7b-v2` 與原始 `Qwen3-ASR-1.7B`，確認 OpenCC / 空白斷句預設有跟著同步。
