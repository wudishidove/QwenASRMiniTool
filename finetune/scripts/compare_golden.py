# -*- coding: utf-8 -*-
"""compare_golden.py — 新產 SRT vs 手校 golden SRT 驗收比對

指標：
1. 專名 occurrence：golden 次數 vs 輸出次數 + 已知錯誤變體（裸吹/激光幕…）殘留數
2. 全文 CER（normalize + 去標點 + t2s 中性化，同 eval.py 計分哲學）
3. 切分健檢：小數被拆（行尾孤數字+行首數字）、詞中切斷樣本
用法：python compare_golden.py --hyp <新srt> [--report <json>]
"""
import sys, json, argparse, re
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from srt_clean import clean_srt
from text_norm import for_cer

GOLDEN_SRT = Path(r"D:\OneDrive\code\yt_download\pokemon_srt") / (
    "誰才是日本第一的裸催王者？最後的決戰，卻逐漸往匪夷所思的方向發展"
    "｜2026 日本全國冠軍賽（二）｜Fiske 講比賽【寶可夢 Champions】"
    "#寶可夢對戰 #nintendoswitch2.srt")

# (正確詞, [已知錯誤變體])；變體殘留應為 0
TERMS = [
    ("裸催", ["裸吹", "螺吹", "裸槌", "裸摔", "裸錘", "裸吹"]),
    ("堅硬腦袋", ["建議腦袋", "金勾臂"]),
    ("急凍鳥", ["極動鳥", "極凍鳥"]),
    ("極光幕", ["激光幕", "巨冠幕"]),
    ("催眠術", []),
    ("妖精氣場", ["妖精氣象"]),
    ("自我再生", ["自我戰勝"]),
    ("破滅之光", ["破滅之拳"]),
    ("一挑二", ["一條人"]),
    ("立大功", []),
    ("濁流", ["鼠王"]),
    ("幽尾玄魚", []),
    ("仆斬將軍", ["仆斬雙刃"]),
    ("雪妖女", []),
    ("風速狗", []),
    ("オオニシ", ["岡尼西", "奧尼西", "奧尼齊", "ウニシ"]),
    ("ヒロシ", ["希路西", "洛西", "ヘイロシ"]),
    ("ホングウ", ["洪古", "洪枯", "紅勾"]),
    ("ショウタ", ["獸打", "秀打"]),
    ("アライ", ["阿拉伊", "艾路雷斯"]),
    ("リンタロウ", ["林達羅", "臉塔洛"]),
]


def cer(ref, hyp):
    try:
        import jiwer
        return jiwer.cer(ref, hyp)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    g_cues = clean_srt(GOLDEN_SRT.read_text(encoding="utf-8"))
    h_cues = clean_srt(Path(args.hyp).read_text(encoding="utf-8", errors="replace"))
    g_all = " ".join(c["text"] for c in g_cues)
    h_all = " ".join(c["text"] for c in h_cues)

    rep = {"terms": {}, "variants": {}}
    print(f"golden cues={len(g_cues)}  hyp cues={len(h_cues)}")
    print(f"\n{'term':<10} golden  hyp   已知錯誤變體殘留")
    worst = []
    for term, variants in TERMS:
        gn, hn = g_all.count(term), h_all.count(term)
        vbad = {v: h_all.count(v) for v in variants if h_all.count(v) > 0}
        rep["terms"][term] = {"golden": gn, "hyp": hn}
        rep["variants"][term] = vbad
        flag = ""
        if gn and hn < gn * 0.9:
            flag = "  ← 低於 90%"
            worst.append(term)
        if vbad:
            flag += f"  變體:{vbad}"
            worst.append(term)
        print(f"{term:<12} {gn:>4} {hn:>5}  {flag}")

    # 全文 CER（計分中性化）
    c = cer(for_cer(g_all), for_cer(h_all))
    rep["cer"] = c
    print(f"\n全文 CER（中性化）: {c:.4f}" if c is not None else "\njiwer 不可用，略過 CER")

    # 切分健檢：hyp 行尾是數字且下一行行首是數字（疑似小數被拆）
    splits = []
    for i in range(len(h_cues) - 1):
        a, b = h_cues[i]["text"].rstrip(), h_cues[i + 1]["text"].lstrip()
        if a and b and a[-1].isdigit() and b[0].isdigit():
            splits.append((h_cues[i]["start"], a[-12:], b[:12]))
    rep["digit_boundary_pairs"] = len(splits)
    print(f"疑似數字跨行拆分: {len(splits)}")
    for s, a, b in splits[:8]:
        print(f"   {s:8.1f}s  …{a} | {b}…")

    # 時間覆蓋斷層（>8s）
    gaps = [(h_cues[i]["end"], h_cues[i + 1]["start"]) for i in range(len(h_cues) - 1)
            if h_cues[i + 1]["start"] - h_cues[i]["end"] > 8.0]
    rep["coverage_gaps_gt8s"] = len(gaps)
    print(f"覆蓋斷層 >8s: {len(gaps)}")
    for a, b in gaps[:8]:
        print(f"   {a:8.1f}s -> {b:8.1f}s ({b-a:.1f}s)")

    print(f"\n重點未達標詞: {sorted(set(worst)) if worst else '無 ✅'}")
    if args.report:
        Path(args.report).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"# report -> {args.report}")


if __name__ == "__main__":
    main()
