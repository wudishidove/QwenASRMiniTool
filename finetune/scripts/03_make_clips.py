"""03_make_clips.py — DATA-6/7/8/9：cue 合併切片 + 正規化 + 語言標記 + manifest/split

流程（每部影片）：
  1. srt_clean.clean_srt 取乾淨 cue
  2. merge_cues：8-12s 視窗（≥8s 累積／>0.6s 間隔或 ≥12s 斷／不切 cue／硬上限 30s）
  3. 依 [首 cue start, 末 cue end] 從 wav16 切 clips/<split>/<clipid>.wav（clipid=整數）
  4. text = text_norm.normalize_text(join cue)；raw_text 保留正規化前
  5. lang：含日文假名→ja，否則 zh
  6. 切分：test=yt5 全部；train 講者(yt1-4)每部尾段 10% → val、其餘 → train
輸出 manifests/{train,val,test}.jsonl。專名校正(對 lexicon)為 DATA-5 後另跑。
"""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from srt_clean import clean_srt
from text_norm import normalize_text, has_japanese_kana, VERSION

DATASET = Path(__file__).resolve().parents[1] / "dataset"
RAW = DATASET / "raw"
WAV16 = DATASET / "wav16"
CLIPS = DATASET / "clips"
MANI = DATASET / "manifests"

TEST_SPEAKER = "yt5"
VAL_TAIL_FRAC = 0.10        # train 影片尾段 10% → val
MIN_S, MAX_S, GAP_S, HARD_S = 8.0, 12.0, 0.6, 30.0
SR = 16000


def merge_cues(cues, min_s=MIN_S, max_s=MAX_S, gap_s=GAP_S, hard_s=HARD_S):
    windows, cur = [], []
    for c in cues:
        if not cur:
            cur = [c]; continue
        gap = c["start"] - cur[-1]["end"]
        cur_span = cur[-1]["end"] - cur[0]["start"]
        span_if = c["end"] - cur[0]["start"]
        if span_if > hard_s:
            windows.append(cur); cur = [c]
        elif cur_span >= min_s and gap > gap_s:
            windows.append(cur); cur = [c]
        elif span_if > max_s and cur_span >= min_s:
            windows.append(cur); cur = [c]
        else:
            cur.append(c)
    if cur:
        windows.append(cur)
    return windows


def find_srt(speaker, vid):
    cand = sorted((RAW / speaker).glob(f"{vid}*.srt"))
    return cand[0] if cand else None


def main():
    import soundfile as sf
    MANI.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        (CLIPS / sp).mkdir(parents=True, exist_ok=True)

    wav_report = json.loads((DATASET / "wav16_report.json").read_text(encoding="utf-8"))
    by_speaker = {}
    for r in wav_report:
        if r.get("status") in ("ok", "exists"):
            by_speaker.setdefault(r["speaker"], []).append(r)

    out = {"train": [], "val": [], "test": []}
    clipid = 0
    stats = {}

    for speaker in sorted(by_speaker):
        for r in by_speaker[speaker]:
            vid = r["id"]
            srt = find_srt(speaker, vid)
            wav = DATASET / r["wav"]
            if srt is None or not wav.exists():
                print(f"[skip] {speaker}/{vid}: srt={srt} wav_exists={wav.exists()}")
                continue
            cues = clean_srt(srt.read_text(encoding="utf-8"))
            windows = merge_cues(cues)

            # 決定該影片每個 window 的 split
            n = len(windows)
            if speaker == TEST_SPEAKER:
                splits = ["test"] * n
            else:
                ntail = max(1, int(round(n * VAL_TAIL_FRAC)))
                splits = ["train"] * (n - ntail) + ["val"] * ntail

            wavdata = None
            for w, split in zip(windows, splits):
                start = w[0]["start"]
                end = w[-1]["end"]
                dur = end - start
                if dur < 1.0:
                    continue
                joined_raw = "".join(c["text"] for c in w)
                # cue 邊界＝創作者每行字幕的天然語句邊界：補全形「，」當邊界標記，
                # 讓微調模型學會在此吐標點，字幕斷句器即可在此切行並隱藏逗號。
                parts = [t for c in w if (t := normalize_text(c["text"]))]
                text = "，".join(parts)
                if not text:
                    continue
                lang = "ja" if has_japanese_kana(text) else "zh"
                # 切音訊
                s0, s1 = int(round(start * SR)), int(round(end * SR))
                data, _ = sf.read(str(wav), start=s0, stop=s1, dtype="float32", always_2d=False)
                if len(data) < SR * 0.5:
                    continue
                cid = f"{clipid:06d}"
                clipid += 1
                rel = f"clips/{split}/{cid}.wav"
                sf.write(str(DATASET / rel), data, SR, subtype="PCM_16")
                out[split].append({
                    "audio": rel,
                    "text": text,
                    "raw_text": joined_raw,
                    "duration": round(len(data) / SR, 3),
                    "lang": lang,
                    "speaker": speaker,
                    "context": "",
                    "video_id": vid,
                    "t0": round(start, 2),
                    "t1": round(end, 2),
                    "norm_version": VERSION,
                })
            stats.setdefault(speaker, {})[vid] = n
            print(f"[ok] {speaker}/{vid}: {len(cues)} cues -> {n} windows")

    for split in ("train", "val", "test"):
        path = MANI / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for o in out[split]:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        durs = [o["duration"] for o in out[split]]
        tot = sum(durs) / 3600
        print(f"# {split}: {len(out[split])} clips, {tot:.2f}h -> {path}")
    print(f"\n# total clips: {clipid}")


if __name__ == "__main__":
    main()
