"""make_v5_split.py — v5：重組 split（新語料 + golden 專名片段；manifest 層）

train.v5 = train.clean(去 TEST_VIDS) ∪ h51REmf4JNY(自 test.clean 移入, 沿 v4)
           ∪ golden.terms.jsonl（手校 golden 含詞視窗，見 make_golden_terms.py）
val.v5   = val.clean(去 TEST_VIDS)
test.v5  = {train,val,test}.clean 中 video_id ∈ TEST_VIDS（與 test.v4 同題目，跨版可比；
           clip id 已因 03 重跑而改變，故依 video_id 重抓而非沿用 test.v4.jsonl）

含 4 道 audio/video disjoint assert + 難詞曝光 grep 自證（不足即中止）。
"""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MANI = Path(__file__).resolve().parents[1] / "dataset" / "manifests"

TRAIN_ADD_VIDS = {"h51REmf4JNY"}
TEST_VIDS = {"Jwy3M2W987c", "VM6LcDCdV1M", "NTt8wMs6OUE"}
GOLDEN_VID = "golden2026jp2"

# 落地前硬性自證：train.v5 內至少要有這些曝光（occurrence 計數）
MIN_OCC = {
    "裸催": 40, "急凍鳥": 5, "堅硬腦袋": 3, "極光幕": 15, "催眠術": 30,
    "破滅之光": 20, "自我再生": 10, "妖精氣場": 8,
    "撒嬌": 10, "戲法空間": 8, "仙子伊布": 5,
    "オオニシ": 5, "アライ": 2,
}


def load(name):
    p = MANI / name
    return [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else []


def occ(rows, term):
    return sum(r.get("text", "").count(term) for r in rows)


def stats(rows, label):
    hrs = sum(r["duration"] for r in rows) / 3600
    vids = sorted({r.get("video_id") for r in rows})
    print(f"{label}: {len(rows)} clips / {hrs:.2f}h / {len(vids)} videos")


def write(rows, name):
    with (MANI / name).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    train_c = load("train.clean.jsonl")
    val_c = load("val.clean.jsonl")
    test_c = load("test.clean.jsonl")
    golden = load("golden.terms.jsonl")
    assert golden, "golden.terms.jsonl 空 — 先跑 make_golden_terms.py"

    train_v5 = [r for r in train_c if r.get("video_id") not in TEST_VIDS]
    train_v5 += [r for r in test_c if r.get("video_id") in TRAIN_ADD_VIDS]
    train_v5 += golden
    val_v5 = [r for r in val_c if r.get("video_id") not in TEST_VIDS]
    test_v5 = [r for r in train_c + val_c + test_c if r.get("video_id") in TEST_VIDS]

    # ── assert 1: audio 三方不重疊 ──
    ta = {r["audio"] for r in train_v5}
    va = {r["audio"] for r in val_v5}
    sa = {r["audio"] for r in test_v5}
    assert not (ta & sa), f"train∩test: {sorted(ta & sa)[:3]}"
    assert not (va & sa), f"val∩test: {sorted(va & sa)[:3]}"
    # ── assert 2: test 不含 golden / h51 ──
    tv = {r.get("video_id") for r in test_v5}
    assert GOLDEN_VID not in tv and not (TRAIN_ADD_VIDS & tv)
    # ── assert 3: test 影片集合正確 ──
    assert tv == TEST_VIDS, f"test.v5 影片集合異常: {tv}"
    # ── assert 4: 難詞曝光自證 ──
    fails = []
    print("== 難詞曝光（train.v5 occurrence）==")
    for t, need in MIN_OCC.items():
        n = occ(train_v5, t)
        mark = "OK" if n >= need else "**不足**"
        print(f"  {t}: {n} (需 ≥{need}) {mark}")
        if n < need:
            fails.append(t)
    assert not fails, f"曝光不足: {fails}"

    print()
    stats(train_v5, "train.v5")
    stats(val_v5, "val.v5")
    stats(test_v5, "test.v5 (凍結題目 = v4 同三部影片)")
    write(train_v5, "train.v5.jsonl")
    write(val_v5, "val.v5.jsonl")
    write(test_v5, "test.v5.jsonl")
    print(f"\n# wrote -> {MANI}\\{{train,val,test}}.v5.jsonl")


if __name__ == "__main__":
    main()
