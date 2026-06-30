"""01_download.py — DATA-1：直接呼叫原生 yt-dlp CLI 下載音訊＋創作者字幕

不走 yt_dlp Python API（CLI 內建多 client 嘗試/格式 fallback/nsig 較穩）。
依 dataset/yt_source.txt（yt1..yt5 為 speaker 標籤）逐部下載：
  yt-dlp -f "bestaudio/best" -x --audio-format m4a \
         --write-subs --no-write-auto-subs \
         --sub-langs "zh-TW,zh-Hant,zh,zh-Hans" --convert-subs srt \
         --no-playlist --download-archive raw/.download_archive.txt \
         -o "raw/<speaker>/%(id)s.%(ext)s" <url>
輸出 raw/<speaker>/<id>.m4a + raw/<speaker>/<id>.<lang>.srt
CLI stdout/stderr 導到 raw/.dl_log.txt（bytes，避免 Windows CJK 解碼崩潰）。

用法： python scripts\01_download.py
"""
from __future__ import annotations
import re, json, subprocess, shutil
from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "dataset"
RAW = DATASET / "raw"
SRC = DATASET / "yt_source.txt"
ARCHIVE = RAW / ".download_archive.txt"
LOG = RAW / ".dl_log.txt"
REPORT = DATASET / "download_report.json"

SUB_LANGS = "zh-TW,zh-Hant,zh,zh-Hans"
AUDIO_EXTS = (".m4a", ".webm", ".opus", ".mp4", ".mp3", ".ogg", ".aac")


def parse_sources():
    speaker = None
    items = []
    for ln in SRC.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if re.fullmatch(r"yt\d+", ln):
            speaker = ln
            continue
        if ln.startswith("http"):
            items.append((speaker, ln))
    return items


def video_id(url: str) -> str:
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{6,})", url)
    return m.group(1) if m else url


def main():
    ytdlp = shutil.which("yt-dlp") or "yt-dlp"
    items = parse_sources()
    RAW.mkdir(parents=True, exist_ok=True)
    report = []
    logf = open(LOG, "ab")
    for sp, url in items:
        outdir = RAW / sp
        outdir.mkdir(parents=True, exist_ok=True)
        cmd = [
            ytdlp,
            "-f", "bestaudio/best",
            "-x", "--audio-format", "m4a",
            "--write-subs", "--no-write-auto-subs",
            "--sub-langs", SUB_LANGS, "--convert-subs", "srt",
            "--no-playlist", "--no-overwrites",
            "--download-archive", str(ARCHIVE),
            "--retries", "5", "--fragment-retries", "5",
            "-o", str(outdir / "%(id)s.%(ext)s"),
            url,
        ]
        logf.write(f"\n\n===== {sp} {url} =====\n".encode())
        logf.flush()
        rc = subprocess.run(cmd, stdout=logf, stderr=logf).returncode
        vid = video_id(url)
        auds = sorted(p.name for p in outdir.glob(f"{vid}.*") if p.suffix.lower() in AUDIO_EXTS)
        srts = sorted(p.name for p in outdir.glob(f"{vid}*.srt"))
        status = "ok" if (auds and srts) else ("no_srt" if auds else ("no_audio" if srts else "fail"))
        report.append({"speaker": sp, "id": vid, "url": url, "returncode": rc,
                       "audio_files": auds, "srt_files": srts, "status": status})
        print(f"[{status}] {sp}/{vid}  rc={rc} audio={auds} srt={srts}", flush=True)
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logf.close()
    ok = sum(1 for r in report if r["status"] == "ok")
    print(f"\n# done: {ok}/{len(report)} ok  -> {REPORT}  (log: {LOG})")


if __name__ == "__main__":
    main()
