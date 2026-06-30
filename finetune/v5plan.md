# Qwen3-ASR 寶可夢 v5 計畫（對抗式生成 + codex 複審定稿）

> 產出方式：3 路 Explore 根因探索 → 13 代理對抗式生成/批判 workflow（逐項 grep 核對語料）→ codex 獨立壓力測試 → 使用者拍板。本檔為唯一推薦路線。

---

## Context（為什麼做、要解什麼）

`pkm-ft-1.7b-v4` 部署後，對一支**全新 Fiske 影片**（2026 日本全國冠軍賽）做推理，光前 3 分鐘就在 `v4實測結果.md` 列出多類錯誤。經對抗分析＋codex＋逐項 grep 語料，**最關鍵的結論是：v4 前 3 分鐘最顯眼的錯誤，絕大多數在 app 推理層，重訓任何模型都救不到**。

- **app 層（重訓無效）**：55 秒整段遺失、9.9 被拆兩半、皮可西→「勒皮可西」、過度斷句/重組。
- **資料/聲學（本輪不修）**：10.9→「十點九」（訓練 target 中文小數＝0 筆，無物可改）、英文 best of 1（train＝0、audio tower 凍結）。
- **訓練訊號弱（v5 可修、且可量測）**：只有「戲法空間／仙子伊布」這一類**已在訓練集但欠曝光**的專名。

**使用者拍板**：①**先修 app（Track B 優先）**；②數字（10.9）**先不管**；③避免打地鼠**先建全 Pokédex 詞表**；④斷句確認是 app 切句造成；⑤已驗證**改用 `start-gpu-official.bat`（固定 5 分鐘切片）可繞過 55 秒黑洞**。

**意圖結果**：先用 app 修補把可見錯誤（尤其 55 秒黑洞）壓掉；再以「全名單 lexicon → 探測集低 recall → 分級過取樣」系統化煉出 `pkm-ft-1.7b-v5`，只對「可靠且可量測」的欠曝光專名負責，不過度承諾。

---

## 一、錯誤歸因表（v4 前 3 分鐘，地面真相已 grep 核對）

語料事實：`train.clean.jsonl`＝5269 筆，其中 **yt1（=Fiske）4467 筆＝84.8%**。

| # | 錯誤 | 根因層級（已驗證） | v5 模型可修？ | 對策（軌） |
|---|---|---|---|---|
| 1 | 整段遺失 ~55s（#104 00:03:33 → #105 00:04:28） | **app-only**。diarize 經查 OFF → `vad` `_detect_speech_groups` 在常門檻漏掉該段語音（非 diar、非 512 截斷）。`:785-787` 空輸出 `continue` 靜默丟段 | ❌ 重訓無效 | **Track B/B6**：VAD gap-fill 補洞（降門檻 re-VAD）+ coverage log + 不靜默丟段 |
| 2a | 10.9→「十點九」 | **base 邊際聲學退化**。實測 train target 中文小數 `X點Y`＝**0 筆**，無 in-domain 訊號可學/可過取樣 | ❌ 幾乎不可修 | **本輪不做**（使用者拍板）；未來只可 app 輸出端後處理 |
| 2b | 9.9 拆成「9」+「9 嗎」 | **app-only**。VAD 在數字中間的微靜音切段 | ❌ | 低優先；數字本輪不做（單調 clamp/gap-fill 不影響此） |
| 3a | 皮可西→「勒皮可西」 | **app-path**。34 clip **全在單一影片 `f0KaoGhjfw8`(=Fiske)**、曝光遠超門檻 → 非欠曝光；「勒」＝切窗/PAD 幻覺。且 `eval.py:47-53` 是 substring，`皮可西 ∈ 勒皮可西`→**算 TP，指標量不到此錯** | ❌ 過取樣無效且反過擬合 | **Track B** 切窗/邊界；**不列模型驗收** |
| 3b | 神秘水滴→「神秘水系」 | **資料缺口**。train **僅 1 clip**（`VEeH7XM42TA`），hold-out 0 → 過取樣＝背單一波形且量不到 | △ 需補來源，本輪不保證 | 標「待補片」，不過取樣 |
| 3c | best of 1→「不要上夥伴」 | 英文術語 train＝0、audio tower 凍結 | ❌ 出界 | 不追 |
| 3d | 戲法空間/仙子伊布類 | **v5 可修**。train.clean 戲法空間=11、仙子伊布=0（需 h51 訊號） | ✅（可量測） | **Track A**：換 base＋平衡 replay |
| 4 | 斷句/合併/重組（#17+18、#36+37、#46） | **多數 app 渲染**（`_split_to_lines`/`_ts_to_subtitle_lines`/`_merge_orphan_lines`、`break_on_space`、`MAX_CHARS=20`）；少數 v2 起 cue 邊界「，」過度斷句（鐵則鎖死不動） | ❌ 模型端 ≈0 | **Track B** 調切行器；cue「，」不動 |
| 5 | 一般詞（比較多→不很多／到底可以→他也可以） | 泛域聲學誤差 | ❌ 難修 | 不設目標 |

---

## 二、Track B — app 修復（**先做、與模型正交、重訓救不了**）

全部在 `D:\OneDrive\code\yt_download\qwen\app-gpu.py`。**先 instrument 診斷再對症，勿盲綁。**

### B0 ✅ 已實測確認（2026-06-30，三版 SRT 對照同一支 Fiske 影片，全長 84.5 分）
| 版本 | 段數 | 覆蓋到 | 大斷層(>8s) | 結論 |
|---|---|---|---|---|
| **VAD**（start-gpu.bat 預設） | 2323 | 01:24:31 | **4 處**（00:03:33→04:28 的 **55.6s** + 32.3s/14.4s/9.5s） | vad coverage hole |
| **1min official** | 2325 | 01:24:31 | **0 處** | ✅ **遺失全解、覆蓋完整** |
| **5min official** | **963** | 01:22:21 | **17 處（150–200s）** | ❌ **嚴重截斷** |
- **遺失有兩個機制（都確認）**：(a) vad 模式 coverage hole；(b) **單段 `transcribe()` 輸出硬上限 `max_new_tokens=512`**（`qwen_asr/inference/qwen3_asr.py`，`generate(..., max_new_tokens=self.max_new_tokens)`）。5min 塊的密集解說遠超 512 token → 每塊吐約 1.5–2.5 分鐘就截斷（17 個斷層恢復點落在 5 分鐘整數倍）→ 丟掉大半內容。
- **使用者「字數超過單一片段輸出上限」的判斷＝正確**。**結論：1 分鐘 official 是甜蜜點**（短到不撞 512 token 上限、又連續覆蓋無 hole）。**5 分鐘不可用。**

### B0b ★ 修正結論（new_1min 實測推翻 official；2026-06-30 晚）
使用者用 official 1min（**已含 B4 停頓斷句**）重轉 = `new_1min`，斷句仍爛。同段 03:10–03:20 對照：

| | VAD | new_1min（official+停頓斷句） |
|---|---|---|
| 斷句 | 「總之 就是非常常見」「大家在排位…」**詞完整** | 「…仆斬將軍的**總**」「**之**就是非常常見**大**」「**家**在排位…」**總之/大家切兩半** |
| 時間重疊段 | 1 / 2323（**0.0%**） | 22 / 2609（**0.8%**） |

- **根因確定**：official 把整條 ~60s 丟給**一次** `transcribe` 再用 FA 時間細分，而 **FA 在長塊上時間戳不可靠**（非單調、22 段重疊）→ B4 停頓斷句吃到噪音時間 → 切在詞中。VAD 在轉錄**前**就用聲學靜音切成短段(~2–10s)→ 每段 FA 穩、斷點落在真實靜音 → 乾淨。**這是 official 的本質缺陷，調切行器救不了。**
- **決策反轉：VAD ＝ 斷句天花板。改「修 VAD 覆蓋（gap-fill）」，不採 official。** diarize 經查為 OFF → 55 秒洞＝**VAD 覆蓋漏掉**（非 diar、非 512 截斷），補洞即可，比馴服 official 簡單且天花板更高。official 降為環境變數選項（覆蓋全但斷句差，不建議）。

### B1（修正）VAD 維持檔案預設；official 降級
- 預設 `segment_mode=vad`（已還原 `_normalize_segment_mode`）。`OFFICIAL_CHUNK_MINUTES_DEFAULT=1.0` 保留（萬一有人用 official 至少不撞 512 截斷）。
- **不再**把 official 設預設、不加 official 為「優化版」；優化改投入 VAD（見 B6）。

### B2 coverage audit log（保留，已實作）
- `:785-787` 空輸出丟段、official 截斷比值、`:813-814/:824-828` except → 印 `[ASR][diag]`（不再靜默）。**全域、不改輸出**，vad/official 都套。

### B3 解吞例外（保留，已實作）
- FA 對齊、孤行合併 `except` 改記 log。

### B4 ★（修正）還原停頓斷句 —— 前提錯誤
- 先前 B4 假設「切行器沒用停頓＝病因」。`new_1min` 實測證明**用 FA 停頓反而吃長塊噪音、切在詞中**（總之/大家被切半）。
- **動作：還原 `_ts_to_subtitle_lines` 的 `pause_break`**（移除該分支，或永遠 False；official 也不要用）。VAD 短段不需停頓斷句——它的斷點本來就在真實靜音。
- `MAX_CHARS`/`break_on_space` 維持原狀（VAD 短段極少觸發硬折）。

### B4c repetition_penalty —— 對 VAD 為次要
- 既已回到 VAD（短段、重複僅 0.1%），repetition 幾乎不發生。**保留 B4c 但僅在 official 模式生效**（已 gate）；VAD 預設路徑維持原樣、不加 penalty。

### B6 ★ 核心修法：VAD gap-fill（補覆蓋、保留 VAD 斷句）+ 單調時間 clamp
- **gap-fill**（`process_file` vad 分支 `app-gpu.py:759-763`）：`_detect_speech_groups` 回傳後算時間軸未覆蓋區間；對每個 gap > ~1.5s：
  1. 先**對該段 audio 以較低門檻 re-VAD**（gap＝常門檻漏掉語音 → 降門檻重切成短段，保 FA 穩、斷點好）；
  2. **保底**：re-VAD 仍空但 gap 仍 >~2s（55 秒洞屬此類，確有語音）→ **以固定 ≤10s 短塊切滿該 gap**（短到 FA 穩、不撞 512），逐塊轉錄，空輸出才丟（B2 log）。
  補出的短段以全域 offset 併回 groups、依時間排序 → **VAD 短段斷句 + 保證全覆蓋**。
  - `_detect_speech_groups` 參數化門檻（現用 global `VAD_THRESHOLD`，`:78/:116`）。diarize 路徑同理補 diar 段間 gap（`:741-753`）。
  - 「保留原始 vad」：偵測到語音的段**分行位元不變**，gap-fill 只「補洞」不改既有斷點 → 符合使用者「保留原始 vad」。
- **單調時間 clamp**：組裝 `all_subs` 後排序，逐段 `start=max(start, prev_end)`、`end=max(end, start+MIN_SUB_SEC)` → 消 FA 噪音造成的重疊（new_1min 22 段 → 0）。**通用、兩模式都套。**
- **驗收**：`start-gpu.bat`（vad + gap-fill）重轉 Fiske → **0 斷層全覆蓋**（55 秒洞補回）+ 斷句維持 VAD 級（無詞中切斷、無「總之/大家」被切半）+ **0 時間重疊**。

### B5 部署 gate（切 v5 必查）
- `_g_opencc_enabled`（`:476`）**預設 True**；切到微調模型必須確認 **OpenCC 關（s2twp 會把岩崩→巖崩、洛托姆→洛託姆改壞）＋ context 空（偏置反傷微調模型）**。把「驗證 app 旗標」寫進步驟。

### B-skip
- **數字後處理（10.9）本輪不做**（使用者拍板）。未來若做，只在 app 輸出端、與 OpenCC 同一 gate、白名單保護含數字專名（十萬伏特/三首惡龍/二連/大比鳥），且守衛高頻誤觸（一點/有點/弱點/N點血/幾點鐘/幾點/努力值 N點）。

---

## 三、Track A — v5 模型（finetune，Track B 修完後）

> 範圍誠實縮小：**換 base 救回被砍訊號 + 全名單驅動的分級過取樣 + 含難詞 val/選點集 + 凍結 test 最終評測**。**配方完全不動。**

### A0 Keystone — 建全 Pokédex 詞表（使用者拍板；anti-打地鼠的前提）
- 現況：`lexicon.tsv` 僅 **21 詞**、`build_lexicon.py` 只有 **18 個種子**、無全 Pokédex 匯入。**沒有這步，「全名單掃描」只是把打地鼠從 `make_v4_replay.TERMS` 搬到 `lexicon.tsv`。**
- 做法：匯入完整 zh-TW **種族/招式/道具/特性**名單（神奇寶貝百科 CSV）→ 擴 `build_lexicon.py` 產出 + 人工 audit 別名/英文保留（Mega/Tera/Z）。
- ⚠️ substring 交叉污染：短名是長名子字串（球/系/伊布 ⊂ 雷伊布/三首…）→ **選點用小而精 hard-word 集；掃描/補資料規劃才用全名單**，掃描器加詞界/去重。

### A1 訓練基底 `train.v5`（避免設計級矛盾）
- ❌ 不可用字面 `train.clean`：`仙子伊布`＝**0**（只在 yt5/Sharlin）→ 會把 v4「仙子伊布 0→5」打回原形＝負優化。
- ❌ 不可用 `train.v4`(=train.yt1+h51)：幾乎純單講者（≈v3）→ 放大跨講者過擬合（v3 教訓：yt5 entity-F −1.9pt）。
- ✅ **定義並新增 `scripts/make_v5_split.py`（additive，仿 `make_v4_split.py`，base 改 train.clean）**：
  ```
  train.v5 = (train.clean 去掉所有 hold-out 影片) ∪ h51REmf4JNY（自 test.clean 移入）
  test.v5  = test.v4（凍結：Jwy3M2W987c / VM6LcDCdV1M / NTt8wMs6OUE）→ 與 test.v4 同題目、CER/entity 可跨版比
  ```
- **codex 必加：4 道 audio-disjoint assert**（現有 `make_v4_split.py:69-71` 只擋 train/test）：
  1. train∩test、val∩test、replay∩test 的 `audio` 無交集；
  2. train∪val∪replay 的 `video_id` ∩ TEST_VIDS＝∅；
  3. `h51REmf4JNY` 不得進 test；
  4. 同 video 的 `(t0,t1)` train/val/選點集不得重疊（片段級洩漏）。
- **落地前硬性 grep 自證並印出**：train.v5 仙子伊布≥7、戲法空間≥11、三首惡龍≥31、撒嬌≥15、皮可西=34；否則中止。

### A2 eval.py 升級 occurrence-level recall（codex 必加 #3）
- 現 `eval.py:47-53` 是 **clip-level substring presence**（一個 clip 出現 5 次只算 1 hit）。「仙子伊布 0→5」是 clip 數、非 recall 分子。
- v5 既以逐詞 recall 當選點標準 → **必須改成比對 prediction/reference 的 occurrence 數量**，否則「一個 clip 喊對就過」門檻太低。

### A3 探測集找低 recall → terms 檔（分級，非逐詞打地鼠）
- 掃描軸＝**「探測集上 recall 低」**（不是「train 內 freq 低」；否則皮可西34、best-of-1=0 都會被誤分類）。
- 分級：
  - **零/單曝光或單一影片**（best of 1=0、神秘水滴=1、皮可西單片）→ **走資料採集（補來源影片重跑 01–04）**，**不過取樣**；
  - **中頻、相異 clip ≥3、跨多影片** → 進 replay 分級過取樣；
  - **高曝光仍錯（皮可西）** → 自動標 **app-path 嫌疑**，不丟過取樣。
- 產出 `terms_v5.tsv`（每列 `term\tR`）。

### A4 replay 詞表參數化 + 護欄
- 新增 `scripts/make_v5_replay.py`（additive，把 `make_v4_replay.py:14` 寫死 `TERMS` 一般化）：讀 `terms_v5.tsv`、`--src train.v5`、substring 比對、**依 audio 去重**、按各詞 R 複本寫單一 `replay.v5.jsonl`（因 `--replay` 只吃一檔、無 `--replay-weight`，per-term 權重只能用複本數模擬）。
- **量化護欄（寫死）**：replay 總列 ≤ train.v5 的 **8–10%**（v4 為 ~4.8%）；**per-term R ≤ 8**；**distinct clip ≥3 才過取樣**（`make_v4_replay` 有 audio 去重但無 clip 數下限，須補 check）；零/單曝光詞一律標「待補來源、不過取樣」。

### A5 含難詞 val + 選點集（codex 必加 #2）
- **`checkpoint-sel.v5`（50–100 列、難詞高密度）**：專供 checkpoint 選擇，**test.v5 全程封存**（避免 v4 的多重比較污染，`結果.md:116-128`）。
- **`val.v5`（≤100–150 列、難詞高密度）**：`train_lora.py:113/135` 的 val loss **只看 seed0 洗牌後前 100 列**，故 val 必須短而密，best-by-val-loss 才有意義（解 v4「val.yt1 無難詞只能採 last」）。

### A6 訓練配方（完全沿用 v2–v4，不引入新變數）
- r16/alpha32/dropout0.05、lr2e-4 cosine、warmup0.05、eff batch 32、3 epoch、bf16/sdpa；LoRA regex 只掛 `thinker` LM、audio tower 凍結；**grad-ckpt 對 `.thinker`**（`train_lora.py:101` 已正確）。
- ⚠️ **訓練嚴禁帶 `--minutes`**：`train_lora.py:47-56/105-108` 對 train+replay 用同一時長預算、且**先 shuffle 再截斷** → replay 出現率不確定、最壞完全消失（codex 強調）。全量訓練。
- `--replay replay.v5.jsonl --val val.v5.jsonl --eval-every 40`。

### A7 checkpoint 選擇（複合準則）
- 對 `epoch0/1/2/last` 各 `merge_lora --skip-gate` → `eval.py --manifest checkpoint-sel.v5 --context off`（用 A2 occurrence recall）。
- 主指標＝**戲法空間/仙子伊布 occurrence recall**；**排除選點**：皮可西（substring 量不到「勒」）、撒嬌/神秘水滴（hold-out=0）。
- 門檻：整體 CER 不比 v4 在 test.v5 退步；hard-word 精集 precision 不掉 >2pt；泛域遺忘（FLEURS）Δ≤+0.03。平手看 micro-F。

### A8 merge + 落地
- `merge_lora.py` merge_and_unload → 重組完整 HF 目錄 → **sanitize generation_config（temperature/top_p/top_k=None）** → `../GPUModel/pkm-ft-1.7b-v5`（**先以 app 下拉可選上線，不改預設**）。
- `05_forgetting_check.py`（FLEURS 外部 Mandarin，乾淨）跑遺忘閘。

### A9 可重現指令（吻合既有參數）
```bash
# venv-train、PYTHONIOENCODING=utf-8
# A0：擴 build_lexicon.py → lexicon.tsv（全 Pokédex + audit）
python scripts/make_v5_split.py            # train.v5/val.v5/checkpoint-sel.v5/test.v5(=test.v4)+4 assert+grep 自證
# A2：升級 eval.py occurrence recall
# A3：探測集找低 recall → terms_v5.tsv
python scripts/make_v5_replay.py --src manifests/train.v5.jsonl --terms terms_v5.tsv --out manifests/replay.v5.jsonl   # 護欄
python scripts/train_lora.py --model ../GPUModel/Qwen3-ASR-1.7B \
  --train manifests/train.v5.jsonl --val manifests/val.v5.jsonl --replay manifests/replay.v5.jsonl \
  --epochs 3 --batch 4 --grad-accum 8 --lr 2e-4 --r 16 --alpha 32 --eval-every 40 \
  --out adapters/p3-1.7b-v5            # 不帶 --minutes
# epoch0/1/2/last 各 merge + eval(checkpoint-sel.v5) 選點 → 採用者 merge 到 GPUModel
python scripts/merge_lora.py --base ../GPUModel/Qwen3-ASR-1.7B --adapter adapters/p3-1.7b-v5/<選中> \
  --out ../GPUModel/pkm-ft-1.7b-v5 --skip-gate
python scripts/eval.py --model ../GPUModel/pkm-ft-1.7b-v5 --manifest manifests/test.v5.jsonl --context off --out reports/B2_1.7b_v5_off.json
python scripts/05_forgetting_check.py --n 40
```

---

## 四、系統化避免打地鼠

1. **Keystone 全名單 lexicon**（A0）：無此步一切退回逐詞。
2. **掃描軸＝探測集 recall 低**（非 train freq 低）。
3. **分級**：零/單曝光→補來源；中頻 distinct≥3→過取樣；高曝光仍錯→app 嫌疑。
4. **substring 污染**：選點用小精集、掃描用全名單，掃描器加詞界去重。

---

## 五、鐵則自評（逐條不違反）

| 鐵則 | 狀態 |
|---|---|
| 別升 transformers | ✅ 4.57.6 |
| target/輸出別做 OpenCC | ✅ 不做；deploy OpenCC 關、context 空 |
| 微調模型別加 context 偏置 | ✅ eval/部署皆 `--context off` |
| 別對 top-level 開 grad-ckpt | ✅ 對 `.thinker` |
| 別寫死 audio_pad | ✅ 動態 |
| cue 串接保留「，」 | ✅ `03_make_clips.py:103-104` 不動；9.9 移交 app official 模式 |
| merge 後 sanitize generation_config | ✅ |
| 不發明 train_lora 參數 | ✅ 只用既有 CLI；過取樣靠 replay 複本數 |
| 數字＝OpenCC 陷阱數字版 | ✅ 整條 target 數字方案 KILL；本輪 app 也不做 |

殘留風險：①train.v5 仍偏 Fiske → 護欄＝test.v5 跨講者不退步 + FLEURS Δ≤+0.03；②難詞只在訓練講者、離線對 Fiske 不可證偽 → 以 Fiske 端到端人工聽測為唯一 GO；③過取樣傾斜 → replay ≤8–10%、R≤8、distinct≥3。

---

## 六、成功準則（兩道閘，事前凍結）

**閘 A — 模型閘（離線、確定性，於封存 `test.v5`，與 v4 同題目）**
- 戲法空間/仙子伊布 occurrence recall ≥ v4；整體 CER 不比 v4 在 test.v5 退步；hard-word 精集 precision 不掉 >2pt；FLEURS 遺忘 Δ≤+0.03。
- **不把 皮可西/撒嬌/神秘水滴/數字 當模型閘**（量不到或無訊號）。

**閘 B — 端到端閘（唯一 GO，於 Track B 修好的 app）**
- app（含 Track B）＋ v5 ＋ OpenCC 關 ＋ context 空，重轉 Fiske 2026 前 3–5 分鐘：
  - (a) **55 秒黑洞消失**（VAD gap-fill；對照時間軸覆蓋率）+ 斷句維持 VAD 級（無詞中切斷）+ 0 時間重疊；
  - (b) 戲法空間/仙子伊布正確、既有專名不退步；
  - (c) **明確標記「仍會錯、不計入模型驗收」**：皮可西「勒」、best of 1、10.9/9.9 → 分開追蹤是否由 Track B 改善。
- **兩閘皆過才考慮把預設由 v2 改指 v5**（v5 先以下拉可選上線）。

---

## 七、被否決的點子（取捨透明）

| 否決/降級 | 原因（實測） |
|---|---|
| 數字 canonicalizer（X點Y→阿拉伯，動 target） | KILL。train 中文小數 0 筆＝死碼；坐落十萬伏特×52/三首惡龍×31 雷區＝全風險零收益 |
| 移除數字間逗號 | KILL。全形數字逗號僅 2 clip 且皆破壞（`1.3，50%`→`1.350%`） |
| cue 邊界數字守衛 | KILL。為近不存在樣式動 v2 刻意「，」設計、破壞合法千分位 |
| base＝字面 train.clean | KILL。仙子伊布=0→負優化 |
| 皮可西過取樣 | KILL。單片足曝光、「勒」＝app、substring 量不到 |
| 神秘水滴 ×N 當「已修」 | 降級。1 clip＝背波形、量不到 → 改「待補來源」 |
| 掃描軸＝freq<K | 替換為「探測集 recall 低」 |
| 「補 Fiske 同講者＝最高槓桿」 | 降級。yt1=Fiske 已 85%，只買聲學多樣性、買不到詞錯/app 修復 |
| val-loss best 當唯一選點 | 替換為封存 test.v5 + 獨立 checkpoint-sel.v5 複合準則 |

---

## 八、端到端驗證

1. **Track B 先**：✅ 已實測（official 1min/5min 因 FA 長塊噪音斷句爛、被推翻；VAD=斷句天花板）→ **修法＝VAD gap-fill 補 55 秒洞（B6）+ 單調時間 clamp**、還原 B4 停頓斷句、保留 B2/B3 log、查部署旗標（B5）。official 降級為環境變數選項。
2. **Track A**：A0 全名單 lexicon → make_v5_split（4 assert+grep 自證）→ eval occurrence 升級 → 探測集 terms_v5 → make_v5_replay（護欄）→ train_lora（不帶 --minutes）→ 四 checkpoint 選點（checkpoint-sel.v5）→ FLEURS 遺忘 → merge+sanitize → `GPUModel/pkm-ft-1.7b-v5`。
3. **兩道閘**：閘 A（封存 test.v5）+ 閘 B（Track-B-修好的 app 重跑 Fiske，唯一 GO）→ 皆過才改預設。

---

## 九、關鍵檔案

- **app（Track B）**：`qwen/app-gpu.py`（VAD `:78/:98-152/:149-150`；official `:80/:554-579/:755-758`；segment_mode env `:598-599`；process_file `:726-844`（空丟 `:785-787`、except `:813-814/:824-828`）；OpenCC 預設 `:476`；official UI `:1155-1171`）；`start-gpu-official.bat`（已存在）/`start-gpu.bat`；`subtitle_lines.py`（`_merge_orphan_lines`）。
- **資料/訓練（Track A）**：`scripts/build_lexicon.py`、`dataset/lexicon.tsv`、新增 `scripts/make_v5_split.py`、新增 `scripts/make_v5_replay.py`、`scripts/eval.py`（occurrence 升級）、`scripts/train_lora.py`（沿用、勿 `--minutes`）、`scripts/merge_lora.py`、`scripts/05_forgetting_check.py`。
- **不動**：`scripts/03_make_clips.py`（cue「，」鐵則）、`scripts/text_norm.py`、`scripts/asr_labels.py`、transformers 4.57.6。
