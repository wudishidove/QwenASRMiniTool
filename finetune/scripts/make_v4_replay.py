"""make_v4_replay.py — V4：過取樣難詞，產生 replay.v4.jsonl

從 train.v4.jsonl 篩出 text 含 {撒嬌, 戲法空間, 仙子伊布} 的 clip（依 audio 去重），
每個 clip 複製 R 份寫成 replay.v4.jsonl。train_lora.py 的 --replay 會把它附加進
訓練集並重洗（既有機制）→ 每詞 clip 每 epoch 曝光 ≈ 1（base）+ R（replay）次。

不修改任何既有腳本。
"""
import sys, json, argparse
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MANI = Path(__file__).resolve().parents[1] / "dataset" / "manifests"
TERMS = ["撒嬌", "戲法空間", "仙子伊布"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="train.v4.jsonl")
    ap.add_argument("--out", default="replay.v4.jsonl")
    ap.add_argument("--r", type=int, default=8, help="每個難詞 clip 複製的份數")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (MANI / args.src).open(encoding="utf-8")]

    # 依 audio 去重挑出含任一難詞的 clip
    seen, picked = set(), []
    for r in rows:
        txt = r.get("text", "")
        if any(t in txt for t in TERMS) and r["audio"] not in seen:
            seen.add(r["audio"])
            picked.append(r)

    # 每詞 clip 數（一個 clip 可能含多詞，分別計）
    per_term = {t: sum(1 for r in picked if t in r.get("text", "")) for t in TERMS}

    dst = MANI / args.out
    with dst.open("w", encoding="utf-8") as f:
        for r in picked:
            for _ in range(args.r):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"== V4 replay (R={args.r}) ==")
    print(f"# 來源 {args.src}: 含難詞 clip = {len(picked)}（已依 audio 去重）")
    for t in TERMS:
        base = per_term[t]
        print(f"#   {t}: {base} clip → 每 epoch ≈ {base * (1 + args.r)} 次曝光（base {base} + replay {base * args.r}）")
    print(f"# replay 列數 = {len(picked) * args.r} → {dst}")


if __name__ == "__main__":
    main()
