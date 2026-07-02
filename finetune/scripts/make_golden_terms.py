"""make_golden_terms.py — v5：從手校 golden SRT+mp3 切出「缺失/低曝光專名」訓練片段

背景（見 v4全片診斷_v5執行.md）：V4 對全新 Fiske 影片手校 diff 顯示「裸催」等詞在
全部訓練語料曝光=0 → 每次必錯且錯法不定，唯一修法＝給訓練訊號。使用者拍板：
接受從 golden sample 取含目標詞的片段回訓（僅含詞窗，全片其餘 ~85% 仍可作端到端驗收）。

流程：mp3 → 16k wav（快取於 wav16/golden/）→ clean_srt → 與 03 相同的 merge_cues
視窗 → 只保留 text 含任一 TERMS 的視窗 → clips/golden_terms/*.wav +
manifests/golden.terms.jsonl（欄位與 03 一致，speaker=yt1、video_id=golden2026jp2）。
"""
import sys, json, importlib.util
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]          # qwen/
sys.path.insert(0, str(ROOT))
from audio_io import load_audio_16k_mono
from srt_clean import clean_srt
from text_norm import normalize_text, has_japanese_kana, VERSION

DATASET = Path(__file__).resolve().parents[1] / "dataset"
CLIPS = DATASET / "clips" / "golden_terms"
MANI = DATASET / "manifests"
WAVDIR = DATASET / "wav16" / "golden"
SR = 16000

GOLDEN_DIR = Path(r"D:\OneDrive\code\yt_download\pokemon_srt")
GOLDEN_STEM = ("誰才是日本第一的裸催王者？最後的決戰，卻逐漸往匪夷所思的方向發展"
               "｜2026 日本全國冠軍賽（二）｜Fiske 講比賽【寶可夢 Champions】"
               "#寶可夢對戰 #nintendoswitch2")
VIDEO_ID = "golden2026jp2"
SPEAKER = "yt1"

# C1 零曝光 + C2 低曝光 + C4 片假名選手名（v4全片診斷 一節）
# v5.1：補 仆斬將軍（v5 端到端 19/24 回退）＋三位漏列選手名（CER 主因之一）
TERMS = [
    "裸催", "堅硬腦袋", "急凍鳥", "立大功",
    "極光幕", "催眠術", "妖精氣場", "自我再生", "破滅之光", "一挑二", "風速狗",
    "仆斬將軍",
    "オオニシ", "ヒロシ", "ホングウ", "ショウタ", "アライ", "リンタロウ",
    "ヤマト", "リュウスケ", "ワタナベ", "コウヘイ", "コバヤシ", "リンヤ",
]

# 與 03_make_clips.merge_cues 相同參數；動態載入避免複製邏輯
_spec = importlib.util.spec_from_file_location(
    "make_clips_03", Path(__file__).resolve().parent / "03_make_clips.py")
_mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mc)
merge_cues = _mc.merge_cues


def main():
    import soundfile as sf
    CLIPS.mkdir(parents=True, exist_ok=True)
    WAVDIR.mkdir(parents=True, exist_ok=True)

    srt_path = GOLDEN_DIR / f"{GOLDEN_STEM}.srt"
    mp3_path = GOLDEN_DIR / f"{GOLDEN_STEM}.mp3"
    assert srt_path.exists() and mp3_path.exists(), "golden SRT/mp3 不存在"

    wav_path = WAVDIR / f"{VIDEO_ID}.wav"
    if not wav_path.exists():
        print("[wav] 轉檔 16k mono …", flush=True)
        data, sr = load_audio_16k_mono(str(mp3_path))
        sf.write(str(wav_path), data, sr, subtype="PCM_16")
    dur_total = sf.info(str(wav_path)).duration
    print(f"[wav] {wav_path.name} {dur_total/60:.1f}min")

    cues = clean_srt(srt_path.read_text(encoding="utf-8"))
    windows = merge_cues(cues)
    print(f"[srt] {len(cues)} cues -> {len(windows)} windows")

    rows = []
    per_term = {t: 0 for t in TERMS}
    cid = 0
    for w in windows:
        parts = [t for c in w if (t := normalize_text(c["text"]))]
        text = "，".join(parts)
        if not text or not any(t in text for t in TERMS):
            continue
        start, end = w[0]["start"], w[-1]["end"]
        dur = end - start
        if dur < 1.0 or dur > 30.0:
            continue
        s0, s1 = int(round(start * SR)), int(round(end * SR))
        data, _ = sf.read(str(wav_path), start=s0, stop=s1,
                          dtype="float32", always_2d=False)
        if len(data) < SR * 0.5:
            continue
        rel = f"clips/golden_terms/g{cid:04d}.wav"
        sf.write(str(DATASET / rel), data, SR, subtype="PCM_16")
        cid += 1
        for t in TERMS:
            if t in text:
                per_term[t] += 1
        rows.append({
            "audio": rel,
            "text": text,
            "raw_text": "".join(c["text"] for c in w),
            "duration": round(len(data) / SR, 3),
            # v5.1：一律 zh — 生產推理永遠用 Chinese token；片假名選手名須在
            # 中文語境下學（v5 曾因 ja 標記把片假名雜訊漏進中文輸出，如フレンじゃんけん）
            "lang": "zh",
            "speaker": SPEAKER,
            "context": "",
            "video_id": VIDEO_ID,
            "t0": round(start, 2),
            "t1": round(end, 2),
            "norm_version": VERSION,
        })

    dst = MANI / "golden.terms.jsonl"
    with dst.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hours = sum(r["duration"] for r in rows) / 3600
    print(f"\n# golden terms: {len(rows)} clips / {hours*60:.1f}min -> {dst}")
    print("# 每詞視窗數（一窗可含多詞）:")
    for t in TERMS:
        occ = sum(r["text"].count(t) for r in rows)
        print(f"#   {t}: {per_term[t]} 窗 / {occ} 次")
    cover = len(rows) / max(1, len(windows))
    print(f"# 佔全片視窗比例: {cover:.1%}（其餘 {1-cover:.0%} 仍為乾淨驗收區）")


if __name__ == "__main__":
    main()
