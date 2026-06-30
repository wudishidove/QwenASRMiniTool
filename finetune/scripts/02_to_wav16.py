"""02_to_wav16.py — DATA-2：raw 音訊 → 16k mono 16-bit PCM WAV

用專案 audio_io.load_audio_16k_mono（零 librosa），輸出 wav16/<speaker>/<id>.wav。
保留 BGM/SFX、不去噪（生產輸入本就有配樂）。可續傳（已存在跳過）。
"""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]          # qwen/
sys.path.insert(0, str(ROOT))
from audio_io import load_audio_16k_mono

DATASET = Path(__file__).resolve().parents[1] / "dataset"
RAW = DATASET / "raw"
WAV16 = DATASET / "wav16"
AUDIO_EXTS = (".m4a", ".webm", ".opus", ".mp4", ".mp3", ".ogg", ".aac")


def main():
    import soundfile as sf
    rows = []
    for spd in sorted(RAW.iterdir()):
        if not spd.is_dir() or spd.name.startswith("."):
            continue
        outdir = WAV16 / spd.name
        outdir.mkdir(parents=True, exist_ok=True)
        for aud in sorted(spd.iterdir()):
            if aud.suffix.lower() not in AUDIO_EXTS:
                continue
            out = outdir / (aud.stem + ".wav")
            if out.exists():
                dur = sf.info(str(out)).duration
                print(f"[skip] {spd.name}/{out.name} ({dur/60:.1f}min)", flush=True)
                rows.append({"speaker": spd.name, "id": aud.stem, "wav": str(out.relative_to(DATASET)),
                             "duration": round(dur, 2), "status": "exists"})
                continue
            try:
                data, sr = load_audio_16k_mono(str(aud))
                sf.write(str(out), data, sr, subtype="PCM_16")
                dur = len(data) / sr
                print(f"[ok]   {spd.name}/{out.name} ({dur/60:.1f}min)", flush=True)
                rows.append({"speaker": spd.name, "id": aud.stem, "wav": str(out.relative_to(DATASET)),
                             "duration": round(dur, 2), "status": "ok"})
            except Exception as e:
                print(f"[err]  {spd.name}/{aud.name}: {str(e)[:120]}", flush=True)
                rows.append({"speaker": spd.name, "id": aud.stem, "status": "error", "error": str(e)[:200]})
    (DATASET / "wav16_report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tot = sum(r.get("duration", 0) for r in rows)
    print(f"\n# {len(rows)} files, total {tot/3600:.2f}h -> wav16/")


if __name__ == "__main__":
    main()
