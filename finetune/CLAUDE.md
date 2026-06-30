# CLAUDE.md — Qwen3-ASR 寶可夢微調（finetune/）

本目錄是把 **Qwen3-ASR-1.7B** 用 **LoRA** 微調成寶可夢對戰專用 ASR 的工作區。
這份是「一眼看懂架構」的索引；**所有細節（數字、踩坑、決策）見 [`結果.md`](結果.md)**。

> 交接全文：[`結果.md`](結果.md)　|　逐步進度與決策：[`todolist.json`](todolist.json)　|　原始計畫：[`dataset/ultracode-finetune-asr-nifty-blum.md`](dataset/ultracode-finetune-asr-nifty-blum.md)

## 現況（一句話）
- **現部署＝`../GPUModel/pkm-ft-1.7b-v2`**（混合 yt1-4，原生繁體＋語句邊界逗號）。
- `../GPUModel/pkm-ft-1.7b-v3`（純 YT1、刻意過擬合）；`../GPUModel/pkm-ft-1.7b-v4`（修 撒嬌/戲法空間/仙子伊布 三詞誤判：重組 split 把 yt5 `h51` 移入訓練＋過取樣 ×8；hold-out 戲法 3→9、仙子 0→5）。**兩者皆非預設**，app 下拉可選。
- 版本沿革／評測數字見 [`結果.md`](結果.md) 的「★ v2／v3／v4 更新」與 §2，報告 JSON 在 [`reports/`](reports)。

## 架構（資料 → 訓練 → 部署）
1. **資料管線** [`scripts/`](scripts)（依序）：`01_download` → `02_to_wav16` → `03_make_clips`（cue 合併 8-12s＋切片＋split：test=yt5、其餘尾段 10%→val）→ `04_qa_gate`（→ `*.clean.jsonl`）。
   來源 [`dataset/yt_source.txt`](dataset/yt_source.txt)；manifest 在 [`dataset/manifests/`](dataset/manifests)。
2. **訓練** [`scripts/train_lora.py`](scripts/train_lora.py)：手寫迴圈，forward 走 `model.thinker(**batch).loss`，LoRA 只掛 thinker LM 層（audio tower 凍結）。labels 手建見 [`scripts/asr_labels.py`](scripts/asr_labels.py)。配方：r16/alpha32/dropout0.05、lr2e-4 cosine、3 epoch、eff batch 32、bf16/sdpa。
3. **合併** [`scripts/merge_lora.py`](scripts/merge_lora.py)：LoRA merge + 重組完整 HF 目錄 → `../GPUModel/pkm-ft-*`。
4. **評測** [`scripts/eval.py`](scripts/eval.py)：yt5 hold-out 的 CER + entity-F1（`--context off|bias`）。
5. **部署**：`../app-gpu.py` 模型下拉自動掃 `../GPUModel/`；切換邏輯／OpenCC／斷句開關詳見 [`結果.md`](結果.md) §8 DEPLOY-1。

可重現指令（v1/v2/v3 全版）見 [`結果.md`](結果.md) §7。

## 環境（別動）
- venv：`venv-train/Scripts/python.exe`（**所有腳本都用它跑**），跑前 `PYTHONIOENCODING=utf-8`。
- **transformers 4.57.6 不可升級**（綁生產推理）。Win10 / RTX 5090 / cu128。

## 「別再踩」精要（完整清單見 [`結果.md`](結果.md) §3、§9）
- ❌ 別對訓練目標/輸出做 **OpenCC**（破壞專名）。❌ 別給微調模型加 **context 偏置**（反傷）。
- ❌ 別對 top-level model 開 grad-ckpt（要對 `.thinker`）。❌ 別寫死 audio_pad 數量（動態）。
- ✅ cue 串接保留邊界逗號（別 `"".join`）。✅ merge 後 sanitize `generation_config`。✅ 部署 = v2 + context 空 + OpenCC 關。
