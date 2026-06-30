"""make_yt1_subset.py — V3：從 clean manifest 篩出「只有 yt1」的訓練/驗證子集

讀 manifests/{train,val}.clean.jsonl，只保留 speaker == "yt1" 的 clip，
分別寫出 manifests/{train,val}.yt1.jsonl，並印出 clip 數與總時長（小時）。
不修改任何既有腳本，亦不動原 clean manifest。
"""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MANI = Path(__file__).resolve().parents[1] / "dataset" / "manifests"
SPEAKER = "yt1"


def subset(src_name, dst_name):
    src = MANI / src_name
    dst = MANI / dst_name
    rows = [json.loads(l) for l in src.open(encoding="utf-8")]
    yt1 = [r for r in rows if r.get("speaker") == SPEAKER]
    with dst.open("w", encoding="utf-8") as f:
        for r in yt1:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = sum(r["duration"] for r in yt1) / 3600
    vids = sorted({r.get("video_id") for r in yt1})
    print(f"# {dst_name}: {len(yt1)}/{len(rows)} clips (yt1), {hours:.2f}h, "
          f"{len(vids)} videos -> {dst}")
    return len(yt1), hours


def main():
    nt, ht = subset("train.clean.jsonl", "train.yt1.jsonl")
    nv, hv = subset("val.clean.jsonl", "val.yt1.jsonl")
    print(f"\n# yt1 total: {nt + nv} clips, {ht + hv:.2f}h "
          f"(train {ht:.2f}h / val {hv:.2f}h)")


if __name__ == "__main__":
    main()
