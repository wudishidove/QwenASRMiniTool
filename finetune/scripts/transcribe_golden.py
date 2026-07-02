# -*- coding: utf-8 -*-
"""transcribe_golden.py — 用指定 GPU 模型走 app-gpu 完整 process_file 路徑
（VAD+gap-fill → transcribe → FA 對齊 → 斷行），對 golden mp3 產出 SRT 供驗收。

用法：venv-train/Scripts/python.exe scripts/transcribe_golden.py --model pkm-ft-1.7b-v5 --out <輸出srt>
部署 gate：OpenCC 關、context 空、break_on_space 關（微調模型帶標點）。
⚠ 需 GPU 空閒（勿與訓練同時跑）。
"""
import sys, argparse, importlib.util, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QWEN = Path(__file__).resolve().parents[2]
GOLDEN_MP3 = Path(r"D:\OneDrive\code\yt_download\pokemon_srt") / (
    "誰才是日本第一的裸催王者？最後的決戰，卻逐漸往匪夷所思的方向發展"
    "｜2026 日本全國冠軍賽（二）｜Fiske 講比賽【寶可夢 Champions】"
    "#寶可夢對戰 #nintendoswitch2.mp3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pkm-ft-1.7b-v5")
    ap.add_argument("--out", required=True, help="輸出 SRT 完整路徑（不可指向 pokemon_srt）")
    args = ap.parse_args()

    out = Path(args.out)
    assert "pokemon_srt" not in str(out), "禁止寫回 pokemon_srt（會覆蓋手校 golden）"
    out.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(QWEN))
    spec = importlib.util.spec_from_file_location("appgpu", QWEN / "app-gpu.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["appgpu"] = m
    spec.loader.exec_module(m)

    # 部署 gate（結果.md §8：微調模型 → OpenCC 關、context 空、break_on_space 關）
    m.ASR_MODEL_NAME = args.model
    m._g_opencc_enabled = False
    m._g_break_on_space = False

    eng = m.GPUASREngine()
    eng.load(device="cuda", cb=lambda s: print(f"[load] {s}", flush=True))
    eng.ready = True

    t0 = time.time()
    last = [-100]
    def cb(i, total, msg):
        if i - last[0] >= 100 or i == total:
            last[0] = i
            print(f"[asr] {msg} ({time.time()-t0:.0f}s)", flush=True)

    # original_path 導向 out 的檔名 → SRT 寫到 out（不動 pokemon_srt）
    fake_ref = out.with_suffix(".mp3")
    res = eng.process_file(GOLDEN_MP3, progress_cb=cb, language=None,
                           context="", original_path=fake_ref)
    assert res is not None, "process_file 回傳 None"
    if res != out:
        res.replace(out)
    print(f"\n# done in {(time.time()-t0)/60:.1f}min -> {out}", flush=True)


if __name__ == "__main__":
    main()
