"""make_v4_split.py — V4：依 video_id 重組 train/val/test split（manifest 層，不重切）

V4 目標：補滿「撒嬌 / 戲法空間 / 仙子伊布」三詞的訓練訊號。
作法（已與使用者確認）：把 yt5 的 h51REmf4JNY 移進訓練集，
其餘兩部 yt5 留作 hold-out，再加 yt4 NTt8wMs6OUE 補測試集時長。

  train.v4.jsonl = train.yt1.jsonl  +  h51REmf4JNY（從 test.clean 取）
  val.v4.jsonl   = val.yt1.jsonl    （不動，僅供監控 val loss）
  test.v4.jsonl  = 其餘兩部 yt5（Jwy3M2W987c, VM6LcDCdV1M） + yt4 NTt8wMs6OUE

clip 實體檔已切好；audio 路徑為相對路徑、與所屬 manifest 無關 → 只重新分配、不搬檔。
不修改任何既有腳本，亦不動原 clean / yt1 manifest。
"""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MANI = Path(__file__).resolve().parents[1] / "dataset" / "manifests"

# ── 重組規則（依 video_id）──
TRAIN_ADD_VIDS = {"h51REmf4JNY"}                         # yt5 → 移進 train
TEST_VIDS = {"Jwy3M2W987c", "VM6LcDCdV1M", "NTt8wMs6OUE"}  # 兩部 yt5 留出 + yt4
TERMS = ["撒嬌", "戲法空間", "仙子伊布"]


def load(name):
    p = MANI / name
    return [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else []


def pick(rows, vids):
    return [r for r in rows if r.get("video_id") in vids]


def stats(rows, label):
    hours = sum(r["duration"] for r in rows) / 3600
    vids = sorted({r.get("video_id") for r in rows})
    tc = {t: sum(1 for r in rows if t in r.get("text", "")) for t in TERMS}
    term_s = " / ".join(f"{t}{tc[t]}" for t in TERMS)
    print(f"# {label}: {len(rows)} clips, {hours:.2f}h, {len(vids)} videos | {term_s}")
    return len(rows), hours


def write(rows, name):
    dst = MANI / name
    with dst.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return dst


def main():
    train_yt1 = load("train.yt1.jsonl")
    val_yt1 = load("val.yt1.jsonl")
    test_clean = load("test.clean.jsonl")
    train_clean = load("train.clean.jsonl")
    val_clean = load("val.clean.jsonl")

    # ── train.v4 = train.yt1 + h51REmf4JNY（在 test.clean 內）──
    h51 = pick(test_clean, TRAIN_ADD_VIDS)
    train_v4 = train_yt1 + h51

    # ── val.v4 = val.yt1（原樣）──
    val_v4 = list(val_yt1)

    # ── test.v4 = 兩部留出 yt5（test.clean）+ yt4（散在 train/val.clean）──
    test_v4 = pick(test_clean, TEST_VIDS) + pick(train_clean, TEST_VIDS) + pick(val_clean, TEST_VIDS)

    # ── 防呆：train 與 test 不可有同一 audio ──
    overlap = {r["audio"] for r in train_v4} & {r["audio"] for r in test_v4}
    assert not overlap, f"train/test 重疊: {sorted(overlap)[:5]}"

    print("== V4 split ==")
    stats(train_v4, "train.v4 (= train.yt1 + h51REmf4JNY)")
    stats(val_v4,   "val.v4   (= val.yt1)")
    stats(test_v4,  "test.v4  (= 2×yt5 留出 + yt4)")

    write(train_v4, "train.v4.jsonl")
    write(val_v4, "val.v4.jsonl")
    write(test_v4, "test.v4.jsonl")
    print(f"\n# wrote -> {MANI}\\{{train,val,test}}.v4.jsonl")


if __name__ == "__main__":
    main()
