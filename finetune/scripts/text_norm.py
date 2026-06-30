"""text_norm.py — 版本化文字正規化（DATA-7 + 評測 ref/hyp 共用）

【設計決定，已用實際資料驗證】
來源字幕本來就是乾淨繁體中文（創作者上傳 zh-TW）。實測 OpenCC：
  - s2twp 詞表會破壞專名：岩崩→巖崩、仆斬將軍→僕斬將軍、洛托姆→洛託姆、類型→型別
  - s2t 字元級也把通用字誤判成簡體（岩/仆/只/吃/才/系/群/托…→冷僻異體）
  - 真正簡體字幾乎不存在（<0.3% 且多為假陽性）
→ 因此**不做繁簡轉換**。文字校正只靠 canonical lexicon（DATA-5）對「已知專名錯拼」
  做保守替換，不做全域 OpenCC。

本檔只保留與繁簡無關、確實需要的正規化：
- normalize_text：全形英數→半形 + 空白收斂（訓練目標＝創作者字幕逐字，保留標點）
- for_cer：normalize_text 後去標點/空白 + 英文小寫（CER/entity-F1 計分，ref/hyp 共用）

【另注意（非本檔職責，需提醒使用者）】
生產 app-gpu.py 對模型輸出套 OpenCC s2twp（:425-426,570），同樣會把
岩崩→巖崩 等專名改壞；微調讓模型輸出正確專名後，該步驟反而會破壞，建議檢討。
"""
from __future__ import annotations
import re

VERSION = "norm-v2-no-opencc"

# 全形英數/空白 → 半形（U+FF01–FF5E → U+0021–007E；U+3000 → 空白）
def _fullwidth_alnum_to_half(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if 0xFF01 <= o <= 0xFF5E:
            out.append(chr(o - 0xFEE0))
        elif o == 0x3000:
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)

# CER：保留 CJK / 假名 / 英數，其餘（標點、空白）去除
_CER_KEEP = re.compile(r"[^0-9A-Za-z㐀-䶿一-鿿぀-ゟ゠-ヿ]")

# 計分用「書寫系統中性化」：對 ref 和 hyp 對稱套 t2s（繁→簡），消除繁簡軸，
# 只測內容/聲學正確度。對稱套用故不偏袒任一方；不影響訓練目標/模型輸出。
_t2s = None
def _to_neutral_script(s: str) -> str:
    global _t2s
    if _t2s is None:
        import opencc
        _t2s = opencc.OpenCC("t2s")
    return _t2s.convert(s)


def normalize_text(s: str) -> str:
    s = _fullwidth_alnum_to_half(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def for_cer(s: str) -> str:
    """計分正規化：t2s 書寫中性化 + 去標點/空白 + 英文小寫。ref/hyp 共用。"""
    s = normalize_text(s)
    s = _to_neutral_script(s)
    s = _CER_KEEP.sub("", s)
    return s.lower()


def has_japanese_kana(s: str) -> bool:
    return bool(re.search(r"[぀-ゟ゠-ヿ]", s))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for t in ["岩崩打化石翼龍 Mega！", "第四隻 仆斬將軍", "洛托姆100％綁講究圍巾"]:
        print(repr(t), "->", repr(normalize_text(t)), "| cer:", repr(for_cer(t)))
