"""srt_clean.py — DATA-3：SRT 解析＋ cue 級文字清理（B 缺口）

清理規則：
  - 去 <...> HTML/font 標籤
  - 去非語音註記：[音樂]/(笑聲)/（掌聲）/【...】/♪…♪ 等
  - cue 內換行併一行
  - 去 emoji / 控制字元
  - 雙語疊行只留含中文的行（去純拉丁字幕行；但保留中文行內的英文術語）
  - 整條空或純註記的 cue → 丟棄
回傳 cues: list[{"idx","start","end","text","raw"}]（秒）。
"""
from __future__ import annotations
import re

# 非語音註記：成對括號內視為註記（[]、()、（）、【】、♪…♪）
_ANNOT = re.compile(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）|【[^】]*】|♪[^♪]*♪|[♪♫]")
_TAG = re.compile(r"<[^>]+>")                       # HTML/font/<c>/<i> 標籤
_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "]+"
)
_CTRL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HAS_CJK = re.compile(r"[一-鿿㐀-䶿]")
_HAS_WORD = re.compile(r"[一-鿿A-Za-z0-9]")
_TS = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{1,3})")


def _ts_to_sec(s: str) -> float:
    m = _TS.search(s)
    if not m:
        return 0.0
    h, mi, se, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000.0


def clean_text(text: str) -> str:
    """清理單一 cue 文字（可能多行）。回傳清理後單行；純註記/空 → ''。"""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    cjk_lines = [ln for ln in lines if _HAS_CJK.search(_TAG.sub("", ln))]
    use = cjk_lines if cjk_lines else lines       # 無中文行則保留原行
    s = " ".join(use)
    s = _TAG.sub("", s)
    s = _ANNOT.sub("", s)
    s = _EMOJI.sub("", s)
    s = _CTRL.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s or not _HAS_WORD.search(s):
        return ""
    return s


def parse_srt(raw: str) -> list[dict]:
    """解析 srt 文字 → 原始 cue（未清理）。容錯 BOM / CRLF / 缺號。"""
    raw = raw.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", raw)
    cues = []
    for b in blocks:
        b = b.strip("\n")
        if not b:
            continue
        lines = b.split("\n")
        ti = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if ti is None:
            continue
        start_s, _, end_s = lines[ti].partition("-->")
        start, end = _ts_to_sec(start_s), _ts_to_sec(end_s)
        text = "\n".join(lines[ti + 1:])
        cues.append({"start": start, "end": end, "raw": text})
    return cues


def clean_srt(raw: str) -> list[dict]:
    """parse + clean，丟棄空/零長 cue。回傳 [{idx,start,end,text,raw}]。"""
    out = []
    for c in parse_srt(raw):
        t = clean_text(c["raw"])
        if not t or c["end"] <= c["start"]:
            continue
        out.append({"idx": len(out), "start": c["start"], "end": c["end"],
                    "text": t, "raw": c["raw"].strip()})
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = sys.argv[1]
    cues = clean_srt(open(path, encoding="utf-8").read())
    dur = cues[-1]["end"] if cues else 0
    print(f"# {path}: {len(cues)} cues, span {dur/60:.1f}min")
    for c in cues[:8]:
        print(f"  [{c['start']:.2f}-{c['end']:.2f}] {c['text']}")
