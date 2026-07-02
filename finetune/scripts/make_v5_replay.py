"""make_v5_replay.py — v5：詞表驅動的分級過取樣 → replay.v5.jsonl

與 make_v4_replay 的差異：per-term R（缺失詞高、次缺詞低）、
一個 clip 命中多詞時取 max R（依 audio 去重、不重複計）、
量化護欄：replay 總列 ≤ train.v5 的 10%（超出即中止）。
"""
import sys, json, argparse
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MANI = Path(__file__).resolve().parents[1] / "dataset" / "manifests"

# per-term 複本數 R（曝光/epoch ≈ base_occ × (1+R)）
TERMS_R = {
    # C1 零曝光（golden 是唯一來源）
    "裸催": 2,          # 53 occ，本已密集，R 小
    "堅硬腦袋": 8, "急凍鳥": 8, "立大功": 8, "一挑二": 8,
    # C2 低曝光（催眠術 base 49 已足；風速狗/破滅之光 v5 端到端已達標 → 退出騰額度）
    "極光幕": 2, "妖精氣場": 3, "自我再生": 2,
    # v5.1：仆斬將軍 v5 端到端 19/24 回退 → 只 anchor golden 影片的視窗（見 VID_ONLY）
    "仆斬將軍": 2,
    # C4 片假名選手名（v5.1 補三位漏列選手）
    "オオニシ": 4, "ヒロシ": 4, "ホングウ": 4, "ショウタ": 4,
    "アライ": 4, "リンタロウ": 4,
    "ヤマト": 4, "リュウスケ": 4, "ワタナベ": 4, "コウヘイ": 4,
    "コバヤシ": 4, "リンヤ": 4,
    # v4 三詞不回退（v4 曝光 144/54/63，R5/5/7 足以持平）
    "撒嬌": 5, "戲法空間": 5, "仙子伊布": 7,
}
CAP_FRAC = 0.10
# 這些詞只從 golden 影片的視窗過取樣（train 內曝光已高、只需 anchor 目標語境）
VID_ONLY = {"仆斬將軍": "golden2026jp2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="train.v5.jsonl")
    ap.add_argument("--out", default="replay.v5.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (MANI / args.src).open(encoding="utf-8")]

    seen = set()
    picked = []          # (row, R)
    per_term_clips = {t: 0 for t in TERMS_R}
    for r in rows:
        txt = r.get("text", "")
        hit = [t for t in TERMS_R if t in txt
               and (t not in VID_ONLY or r.get("video_id") == VID_ONLY[t])]
        if not hit or r["audio"] in seen:
            continue
        seen.add(r["audio"])
        for t in hit:
            per_term_clips[t] += 1
        picked.append((r, max(TERMS_R[t] for t in hit)))

    total = sum(R for _r, R in picked)
    cap = int(len(rows) * CAP_FRAC)
    print(f"== replay.v5: {len(picked)} distinct clips -> {total} 列（cap {cap}）==")
    for t, R in TERMS_R.items():
        base = sum(r.get("text", "").count(t) for r in rows)
        rep = sum(r.get("text", "").count(t) * Rr for r, Rr in picked)
        rep_t = sum(r.get("text", "").count(t) * Rr for r, Rr in picked if t in r.get("text", ""))
        print(f"  {t}: clips={per_term_clips[t]} base_occ={base} replay_occ={rep_t} "
              f"曝光/epoch≈{base + rep_t}")
    assert total <= cap, f"replay {total} 列超過 train 10% 護欄（{cap}）— 調低 R"

    with (MANI / args.out).open("w", encoding="utf-8") as f:
        for r, R in picked:
            for _ in range(R):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n# wrote {total} rows -> {MANI / args.out}")


if __name__ == "__main__":
    main()
