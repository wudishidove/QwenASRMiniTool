# 實作計畫：微調 Qwen3-ASR 用於寶可夢對戰語音辨識

## Context（為什麼做這件事）

使用者的桌面字幕工具（`start-gpu-official.bat` → `app-gpu.py`，PyTorch CUDA + `qwen-asr` 套件 + `GPUModel\Qwen3-ASR-1.7B`）在「寶可夢對戰講解」這個窄域辨識**誤判偏高**，尤其專有名詞（種族／招式／道具／特性）與繁中語域。現行靠 `context=`（`pokemon.txt`）做情境偏置，但**效果不足**，且 **1.7B 塞不下太長的 prompt**（無法把上千詞庫全餵進去）。

**目標結果**：在不破壞既有推理／字幕管線的前提下，微調 Qwen3-ASR，把**高頻專名 + 講解語域 + 繁中輸出**烤進權重，降低誤判；長尾詞彙仍以 scoped 偏置輔助。產出的權重需 **drop-in** 回 `app-gpu.py`（換一個值即上線／秒退回滾）。

### 本機已驗證事實（決定整體策略，非推測）
- `transformers 4.57.6`：`from transformers import Qwen3ASRForConditionalGeneration` → **ImportError**（無 native `qwen3_asr`）。升級 transformers 會動到綁 4.57.6 的**生產推理路徑**，風險高 → **不升級**。
- **`qwen_asr` 套件自帶可訓練模型碼**（codex 已對 file:line 驗證）：`Qwen3ASRThinkerForConditionalGeneration.forward(labels=...)` 回傳 `.loss`（`modeling_qwen3_asr.py:1243-1250`）、`supports_gradient_checkpointing=True`（`:1292-1297`）。→ **不升 transformers、不下 `-hf` 權重即可訓練**，輸出格式與磁碟 non-hf checkpoint 一致 → merge 後直接 drop-in。
- GPU = **RTX 5090 / ~32GB / sm_120**，torch 2.7.0+cu128 → **VRAM 非限制**，不需 WSL2／DeepSpeed／QLoRA。
- 唯一寶可夢配對語料 = 單講者 ~2h15m 機器字幕影片；`pokemon.txt` 僅 ~30 詞。**使用者已確認可再收 10h+、多講者**。

### 使用者已拍板
1. **語料**：已備 **~7h / 3 講者 / SRT 確定準確**（`finetune\dataset\yt_source.txt`，yt1 5 部＋yt2 2 部＋yt3 1 部），可再收至 10h+ → 可做 speaker-disjoint、可宣稱泛化、最終可上 1.7B。
2. **canonical 拼寫**：**中文為主、英文術語保留**（噴火龍／十萬伏特＝中文；Mega／Tera／Z 招等＝英文）。
3. **分工**：**微調為主、偏置為輔**。

---

## 推薦路徑（單一方案）

**0.6B LoRA 煙霧測試 → 3–5h 首個真實 LoRA（speaker-disjoint 評測）→ go/no-go → 10h+／1.7B 擴大**。
訓練棧用 **vendored `qwen_asr` 模型碼 + PEFT LoRA**，LoRA 只掛 **thinker 文字層**（凍結 audio encoder＋projector＋embedding、不動 vocab → ForcedAligner 不受影響）。偏置為輔：把 `pokemon.txt` 升級成 scoped／結構化／canonical 詞表，推理時與微調並用。

| 階段 | 做法 | 退出條件 |
|---|---|---|
| **Phase 0 偏置基準（零訓練、先做）** | 現有 1.7B 對小評測集跑「偏置 OFF/ON」兩版，量 CER＋entity-F1 → 得到 **B1（要打敗的真 floor）** | 產出 B0/B1 數字表 |
| **Phase 1 0.6B LoRA 煙霧（半天）** | vendored modeling＋PEFT，下 `Qwen/Qwen3-ASR-0.6B`(non-hf)，用現有影片 20–30 分 | M1 gate（見里程碑）全過才前進 |
| **Phase 2 首個真實 LoRA** | 3–5h 校正 in-domain＋通用 replay，推理開 scoped 偏置，speaker-disjoint 術語測試集評測 | M2 go/no-go |
| **Phase 3（條件觸發）** | 同配方上 `Qwen3-ASR-1.7B` 或收至 10–50h | 僅當 M2 證明擴模型／擴資料確有收益 |

---

## 使用者必須提供的資料（本計畫重點）

> **已確認的實際資料集**：`qwen\finetune\dataset\yt_source.txt` 列了 **5 組來源（yt1=6 部、yt2=3 部、yt3=1 部，共 12 部影片），~7h、5 講者，SRT 確定準確**。這**足以走完 Phase 1 + Phase 2**（含 speaker-disjoint 評測）並決定是否上 1.7B，逼近 Phase 3 下限。下表是分階段餵入量；不用一次到位。
> **5 講者切分（hold-one-speaker-out）**：抽**最小一組（很可能 yt5）整個當 test**（術語測試集，需 ≥30–60 分），另 4 組當 train（每組 ≥1h）。
> **SRT 來源＝創作者上傳的官方字幕**（非自動字幕）→ 下載指定創作者字幕軌；品質佳但仍須清 HTML/註記/換行，且 3 個頻道專名拼寫/中英混用很可能不一致 → 全部過 canonical lexicon（中文為主、英文術語保留）+ OpenCC `s2twp`，並抽查專名位置。

| 階段 | in-domain 音訊（時長 / 段數 8–12s） | 講者 | 通用 replay | 評測集 | 標註工時（bootstrap＋校正 ≈2–4×RT） |
|---|---|---|---|---|---|
| **Phase 0** | 0（用現有影片切片） | — | 0 | 手工校正 **10–15 分 ≈ 80–120 段** ＋ canonical 詞表 **~50–150 條（實際出場詞）** | ~0.5–1 人天 |
| **Phase 1 煙霧** | **20–30 分 ≈ 150–250 段**（取自 yt1 任一部即可） | 1 可 | 0 | held-out **5 分** | ~1–2 人天 |
| **Phase 2 首個 LoRA** | **train ~5–6h（yt1+yt2，2 講者）** | **3 已備**（yt1/yt2/yt3） | **3–10h 通用 zh**（in:general ≈ 1:1–1:2，來源見下） | **yt3 整組 hold-out 當 speaker-disjoint 術語測試集（≥30–60 分 / 目標 200–500 句、每句 ≥1 專名）** | ~6–20 人時（SRT 已準，主要做 canonical 正規化＋抽查） |
| **Phase 3（條件）** | **擴至 10–50h ≈ 5k–40k 段**（需再收影片/講者） | 4+ 更佳 | 等比 replay | 擴充術語測試集 | ~40–80+ 人時 |

**為何是這些數字**：窄域 CER 膝點在前 ~5h（每標註小時收益最大）；5→50h 銳化專名與講者穩健度但邊際遞減。**專名準確率由「該詞是否出現、是否 ≥10–30 次跨 ≥3–5 講者」決定，比原始時數更關鍵**；1–2 次出現的名字音訊學不會，靠偏置補。

**通用 replay（首跑即納入）**：Phase 2 首跑混 **3–5h 通用 zh**，首選 **Common Voice zh-TW（HF `datasets`，已 16k、零標註成本）**，不夠再加 AISHELL-1/2 或 `download_video/` 校正段；in:general ≈ 1:1–1:2，獨立 `replay.jsonl`、正規化同套。理由：生產 app 也做通用轉錄，遺忘代價直接傷日常使用，replay 是最便宜的保險。

### 資料前處理執行規格（yt_source.txt → 訓練就緒 manifest）

**目錄結構**（一律 `qwen\finetune\dataset\`）：
```
dataset/
  yt_source.txt                 # 來源清單（yt1/yt2/yt3）
  raw/<speaker>/<id>.{m4a,srt}   # 下載原始音訊＋官方字幕
  wav16/<speaker>/<id>.wav       # 16k mono 全長
  clips/<split>/<clipid>.wav     # 切片；clipid = 純整數（避免 Windows CJK 檔名 mojibake）
  lexicon.tsv                    # canonical / 別名 / 類別 / 是否保留英文
  manifests/{train,val,test,replay}.jsonl
```

**步驟**：
1. **下載官方字幕軌**：yt-dlp 抓最佳音訊 + **創作者字幕**（`--write-subs --sub-langs "zh-Hant,zh-TW,zh" --no-write-auto-subs`；缺則退 zh-Hans），依 yt1/yt2/yt3 標 `speaker`，記 video id／時長。可重用專案 `downloader.py` 的 SSL/續傳。
2. **轉 16k mono**：`audio_io.load_audio_16k_mono` → `wav16/`。
3. **SRT 解析＋文字清理（cue 級，B 缺口）**：去 `<...>` HTML/font 標籤、去 `[音樂]/(笑聲)/♪/【】` 類非語音註記、cue 內換行併一行、去 speaker 標籤/emoji；**整條空或純註記的 cue 丟棄**；雙語疊行只留中文。
4. **對齊驗證（A 缺口）**：每部抽查 5–10 條 cue 聽對齊、偵測整體偏移（片頭/廣告）；必要時用現成 **`Qwen3-ForcedAligner` 對「清理後文字＋全長音訊」重對齊**取乾淨詞級時間（比原 cue 更準）。偏移無法修者整部剔除並記錄。
5. **建 canonical lexicon（C 缺口，前置子任務）**：以全寶可夢 zh-TW 名單（種族/招式/道具/特性，來源：神奇寶貝百科/wiki 匯出 CSV）為底 + 從 8 部清理字幕抽高頻候選 → 人工定 `lexicon.tsv`（canonical 中文為主、別名/異拼、保留英文術語 Mega/Tera/Z）；`pokemon.txt` 併入擴充。
6. **cue 合併切片**：清理後 cue 依 8–12s 視窗合併（≥8s 累積／>0.6s 間隔或 ≥12s 斷／不切 cue 中間／**硬上限 30s**），依 `[首 cue start, 末 cue end]`（或重對齊時間）切 `clips/`，clipid 用整數。
7. **正規化＋專名校正**：`text` 套版本化正規化（OpenCC `s2twp` → 半/全形 → 標點保留）；**專名位置一律對 `lexicon.tsv` 校正**（官方字幕跨頻道仍可能不一致）；保留 `raw_text`。
8. **語言標記**：以 `zh` 為主，偵測到日文段才標 `ja`。
9. **輸出 manifest＋分割**（見「切分與術語測試集」）。
10. **QA 驗證閘（G 缺口，訓練前必過）**：腳本逐筆檢查 audio 存在、sr=16k mono、`duration` 與檔案相符、`text` 非空、**無段 >30s**、`lang∈{zh,ja}`、**手建 labels 流程對該段 tokenize round-trip 成功且 `(labels!=-100).sum()>0`**；任一失敗列報、不入訓練集。

### 音訊格式規格（一次定死，train/val/test/CER 同套）
- **16 kHz、mono、16-bit PCM WAV**（或 FLAC）。一次性 resample；既有 `audio_io.load_audio_16k_mono` 正好產這格式。
- **保留 BGM/SFX、不去噪**（生產輸入本就有配樂）。
- **分段 5–20s（目標 8–12s），絕不切在字中間**，切在 VAD 靜音；可重用 `app-gpu.py` Silero VAD 邊界（訓練/推理一致）。
- 既有 SRT cue 是 0.6–3s 字幕行 → **合併成 8–12s 視窗**（累積 ≥8s；遇 >0.6s 間隔或 ≥12s 斷開；不切 cue 中間），依 `[首 cue start, 末 cue end]` 切 WAV。

### Manifest（JSONL，一行一物件，字面範例）
```json
{"audio": "clips/0001.wav", "text": "花葉蒂使出魔法葉", "raw_text": "花葉蒂使出魔法葉", "duration": 9.4, "lang": "zh", "speaker": "showA_caster1", "context": "花葉蒂 魔法葉 阿羅拉九尾 暴風雪"}
```
`audio`/`text`(最終正規化目標)/`duration`/`lang`(zh/ja 逐段標)/`speaker`(split 用) 必填；`raw_text` 保留正規化前原文；`context` = 該段 scoped 詞表（這場實際出場，非全部）。

### 文字正規化（決定一次、寫成版本化函式、ref/hyp 共用）
1. **繁中輸出**：reference 跑 OpenCC **`s2twp`**（與 `app-gpu.py:425-426,570` 一致），讓模型原生輸出 zh-TW。
2. **半/全形**：ASCII 數字半形、CJK 標點全形，統一。
3. **canonical lexicon（最重要，依使用者政策＝中文為主／英文術語保留）**：每個專名唯一寫法（噴火龍／十萬伏特；Mega／Tera 保留英文）；**專名位置一律人工對此表校正**。
4. **標點**：`text` 保留（字幕需要），**CER 計分時剝除**。

### 切分與術語測試集（D/E 缺口已補）
- **test = yt3 整組**（speaker-disjoint，train 不得見此講者）。
- **val = yt1+yt2 各部時間尾段**（每部最後 ~10%，in-distribution，供 early-stop）；其餘 yt1+yt2 為 **train**。
- **術語測試集** = test(yt3) 中**正規化後含 ≥1 個 `lexicon.tsv` 詞**的段，目標 200–500 句、跨種族/招式/道具/特性；報 **entity-recall/F1/term-error**，與整體 CER 分開。
  - **yt3 句數不足 fallback**：另從 train 講者抽含專名句記為 `term_test_indist`（標明**非** speaker-disjoint），與 yt3 集**分開報**，不混淆泛化結論。
- **通用域 zh test**（取 Common Voice zh-TW held-out 一小段）監控災難性遺忘。

---

## 環境與訓練（已併入 codex 修正）

### 套件（獨立訓練 venv，不污染生產 `venv-gpu`）
沿用 torch 2.7.0+cu128，加 `peft`、`accelerate`、`datasets`、`soundfile`、`soxr`、`jiwer`、`opencc-python-reimplemented`。**不升級 transformers**。下載 `Qwen/Qwen3-ASR-0.6B`(non-hf)。

### 訓練配方（codex 修正後）
- **Loss**：causal-LM cross-entropy，只算轉錄 token（其餘 -100）。**凍結 audio encoder＋projector＋embedding，LoRA 只掛 thinker 文字層**，r=8(煙霧)/16(首跑)、alpha=2r、dropout 0.05–0.1、LR 1e-4–2e-4、warmup 5%、≤3 epoch、**early-stop 看 val CER＋entity-F1**、replay 1:1–1:2。

- **【BLOCKER 修正 1】訓練入口走 `model.thinker`**：top-level `Qwen3ASRForConditionalGeneration` class body 只有 `generate()`；loss 的 `forward()` 在 `model.thinker`（`modeling_qwen3_asr.py:1159-1250`）。→ 訓練 wrapper **直接呼叫 `model.thinker(input_ids=..., labels=...).loss`**，或 subclass override top-level forward 委派到 thinker；不能裸丟 `Trainer(model=top_level)`。

- **【BLOCKER 修正 2】target_modules 用精確 regex 避開 audio tower**（audio tower `:472` 也有同名 `q/k/v_proj`）：
  ```
  target_modules = r"thinker\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
  ```
  訓練前 `model.print_trainable_parameters()` 確認只命中 thinker LM、audio tower 0 個。

- **【ERROR 修正 3】labels 動態建構，不寫死 390**：`390×audio_pad` 只對 30 秒音訊成立（`generate_prompt_template.py:41-43` 用 480000 samples 產生）。→ **對每段呼叫 processor 取 `input_ids`，在 `input_ids` 中動態定位 `audio_pad`(151676) 區段**，再 mask。

- **【ERROR 修正 4】手建 labels 為唯一路徑**：`processing_qwen3_asr.py` 的 `__call__` 是 pass-through、chat_template 無 `{% generation %}` block，**沒有 `apply_chat_template(output_labels=True)`**。依 `prompt_template.json` 手組：`prefix_ids` + (N×`audio_pad`151676) + `suffix_ids`[151670,151645,198,151644,77091,198] + Chinese `language_suffix`[11528,8453,151704] + 轉錄 token + eos 151645，**只監督「language_suffix＋轉錄＋eos」，其餘 -100**。
  - **collator 規格（codex 缺口）**：batch padding `input_ids/attention_mask/input_features/feature_attention_mask/labels`；**labels 長度必須 == input_ids 長度、padding 填 -100**。
  - **斷言（安全網）**：`labels.shape == input_ids.shape` 且 `(labels!=-100).sum() == 轉錄＋尾巴 token 數`。

- **【缺口】backward smoke test**：跑任何 epoch 前先確認 loss 非 NaN、LoRA param 有 grad、audio_tower param `.grad is None`、LoRA patch 後 top-level `generate()` 仍可走；20–30 句過擬合到 loss≈0。

### 代表性指令骨架
```python
import torch
from peft import LoraConfig, get_peft_model
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from transformers import TrainingArguments, Trainer

MODEL = r"D:\models\Qwen3-ASR-0.6B"      # non-hf
model = Qwen3ASRForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")  # FA2 可用時改 flash_attention_2
for p in model.parameters(): p.requires_grad = False
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=r"thinker\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"))
model.print_trainable_parameters()       # 驗證只命中 thinker LM
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
# 注意：Trainer 需確保 forward 路由到 thinker 取 loss（見 BLOCKER 修正 1）
```

---

## 評測

- **工具**：`jiwer` WER ＋ 字元級 CER（CJK 去空白、字元切分）。解碼後剝 `language<asr_text>` 標籤。
- **正規化（ref/hyp 同一版本化函式）**：OpenCC `s2twp` → 去標點 → 去空白 → 字元切分。
- **三 baseline 並排**：**B0**=base 偏置 OFF；**B1**=base 偏置 ON（scoped）= **真 floor／現行 1.7B＋偏置**；**B2**=base＋LoRA 偏置 ON。
- **三組指標**：(1) 整體 CER/WER；(2) **術語 entity-F1/term-error**；(3) 通用域 CER（遺忘監控）。
- **entity-F1 計算規格（codex 缺口）**：以 canonical lexicon（OpenCC 正規化後）做匹配；逐句抽出 ref/hyp 命中的專名集合 → per-term exact match → macro entity-F1＋entity-recall。**`pokemon.txt` 逗號串不可直接當 eval 詞表**，先轉成 canonical TSV（term／別名／繁中形式）。

---

## 整合回管線

- **部署＝換權重**：LoRA `merge_and_unload()` → **【ERROR 修正 5】重組完整 top-level HF 目錄**（複製原始 `Qwen3-ASR-*` 目錄，把 thinker 的 merged safetensors 覆寫進去；**保留** `config.json`(model_type=qwen3_asr)、`preprocessor_config.json`、`chat_template.json`、`generation_config.json`、特殊 token、`model.safetensors.index.json`）→ 放 **`GPUModel\pkm-ft-vN\`（不覆寫原始）**。
- **切換**：改 `ASR_MODEL_NAME`（`app-gpu.py:63`）或 `model_tab.py` 路徑，**改一值即上/秒退**。
- **載入相容性閘（每次產模型必過，codex 缺口補腳本）**：`qwen_asr.Qwen3ASRModel.from_pretrained(新目錄)` → 3 段 `transcribe()` 出字 → `parse_asr_output()` 不報錯。
- **ForcedAligner 不動**（不改 tokenizer/vocab）→ 詞級 SRT 時間軸照舊。
- **caveat**：faster-whisper/CTranslate2（舊 `../src/`）與 CPU 版 `app.py`（OpenVINO/GGUF/CrispASR）**吃不了此 HF 微調**；整合對象只有 PyTorch 路徑（要享受到須各自轉檔）。

---

## 里程碑與 Go/No-Go 閘

- **M0（Phase 0）**：產出 B0/B1 的 CER＋entity-F1 表。
- **M1（Phase 1 煙霧）pass＝全部成立**：loss 單調下降 ∧ 30 句過擬合 loss≈0 ∧ LoRA 有 grad/audio_tower 無 grad ∧ held-out CER 算得出 ∧ **merge 後完整目錄經 `app-gpu.py`/`qwen_asr` 載入且 `parse_asr_output` 正常**。**Fail → 修接線，不前進。**
- **M2（Phase 2，全程偏置 ON、speaker-disjoint）**：
  - **No-Go to 1.7B／直接出 0.6B**：B2 已達標。
  - **Go to 1.7B**：B2 明顯贏 B1（CER −≥2 pts ∧ entity-F1 +≥3 pts）但仍未達標 → 同配方上 1.7B。
  - **STOP（別擴模型，去補詞表/資料）**：B2 對 entity-F1 提升 < 2–3 pts over B1 → 瓶頸是詞彙覆蓋/偏置品質。
  - **遺忘守門**：通用 zh 控制集明顯退化 → replay 提至 1:2–1:3 / 降 LR，未過不前進。
- **M3（Phase 3 條件觸發）**：speaker-robust 域內準確率＋術語達標＋過全套相容/A-B。

---

## 風險與緩解

| 風險 | 緩解 | 回滾 |
|---|---|---|
| 訓練掛錯層級（loss 在 thinker、target 誤打 audio tower） | 走 `model.thinker` forward＋精確 regex＋`print_trainable_parameters` 驗證 | — |
| labels 寫死 390 / 手建易錯 | 動態定位 audio_pad＋labels 長度斷言＋過擬合測試 | 修 collate 不上線 |
| 機器字幕標籤雜訊與目標相關（最陰險） | canonical lexicon＋專名強制人工校正＋用 baseline 跑全片、CER>40% 句標疑似雜訊複審 | 重標該批 |
| 小量音訊學不會專名 | 微調修高頻＋scoped 偏置補長尾；先量 entity-recall 再決定擴量 | `context=""` 還原 |
| merge 後生產載不動 | 重組**完整** top-level 目錄＋載入相容性閘腳本 | 退原始 `GPUModel\Qwen3-ASR-1.7B\` |
| 災難遺忘/過擬合 | LoRA r8–16、≤3 epoch、early-stop on val CER、1:1–1:3 replay | 退舊權重 |
| 升級 transformers 弄壞生產 app | 走 vendored 碼、不升 transformers、獨立訓練 venv | 生產 venv 不動 |
| 偏置清單膨脹反退化（2× WER） | scoped＋結構化包裝（「寶可夢：…；招式：…」），勿平鋪 | `context=""` 還原 |

---

## 驗證（端到端如何測）

1. **管線接線**：`python` 跑 backward smoke test（20–30 句）→ 斷言 loss 下降、LoRA 有 grad/audio_tower 無 grad、labels 長度＝input_ids 長度。
2. **煙霧過擬合**：同 30 句訓到 loss≈0；held-out 5 分算出 CER（值不重要，證明流程通）。
3. **merge＋載入閘**：merge → 重組完整目錄 → `qwen_asr.Qwen3ASRModel.from_pretrained(新目錄)` → 3 段 `transcribe()` → `parse_asr_output()` 無例外。
4. **生產 A/B**：固定 held-out 影片，`app-gpu.py` 分別指向原始 vs `pkm-ft-vN`，比整體 CER／entity-F1＋人工抽查字幕分行；通過才把預設指向新模型，舊目錄留回滾。
5. **遺忘檢查**：通用 zh test 集 CER 不顯著退化。

---

## 關鍵檔案
- 整合目標/載入器：`D:\OneDrive\code\yt_download\qwen\app-gpu.py`（`ASR_MODEL_NAME` :63；`from_pretrained` :541；`transcribe(context=)` :583-598；OpenCC :425-426,570）
- vendored 可訓練模型碼：`...\Python312\Lib\site-packages\qwen_asr\core\transformers_backend\{modeling_qwen3_asr.py,processing_qwen3_asr.py,configuration_qwen3_asr.py}`（thinker forward :1159-1250、grad ckpt :1292-1297、audio tower 同名層 :472）
- 標籤格式真相：`D:\OneDrive\code\yt_download\qwen\prompt_template.json`（zh 尾巴 [11528,8453,151704]、asr_text_id 151704、audio_pad 151676）
- 偏置詞表（需升級成 canonical TSV）：`D:\OneDrive\code\yt_download\qwen\pokemon.txt`
- **主資料集來源清單**：`D:\OneDrive\code\yt_download\qwen\finetune\dataset\yt_source.txt`（yt1/yt2/yt3＝5 講者、11 部、~7h、準確 SRT；需先用 yt-dlp 下載音訊＋字幕並轉 16k mono）
- 模型落點：新建 `D:\OneDrive\code\yt_download\qwen\GPUModel\pkm-ft-vN\`（不覆寫原始 `Qwen3-ASR-1.7B\`）
- aligner（不動）：`...\qwen\GPUModel\Qwen3-ForcedAligner-0.6B\`
- 音訊載入/切片：`D:\OneDrive\code\yt_download\qwen\audio_io.py`
