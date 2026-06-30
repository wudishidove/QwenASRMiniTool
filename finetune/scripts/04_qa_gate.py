"""04_qa_gate.py — DATA-10：訓練前 QA 驗證閘（G 缺口）

逐筆檢查每個 manifest clip：
  - audio 檔存在
  - sr=16000、mono、subtype 為 PCM
  - duration 與檔案實際長度相符（±0.15s）
  - text 非空
  - 無段 >30s（且 >= 0.5s）
  - lang ∈ {zh, ja}
  - 手建 labels round-trip：tokenizer.encode(text) 非空 → (labels!=-100).sum()>0
任一失敗 → 列報並從 *.clean.jsonl 剔除；全通過則 clean == 原檔。
"""
import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
MODEL = str(ROOT / "GPUModel" / "Qwen3-ASR-0.6B")
DATASET = Path(__file__).resolve().parents[1] / "dataset"
MANI = DATASET / "manifests"
MAX_S, MIN_S = 30.0, 0.5


def load_tokenizer():
    import qwen_asr.inference.qwen3_asr  # 觸發 AutoProcessor 註冊
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL, fix_mistral_regex=True)
    return proc.tokenizer


def main():
    import soundfile as sf
    tok = load_tokenizer()
    grand_fail = 0
    for split in ("train", "val", "test"):
        path = MANI / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        good, fails = [], []
        for r in rows:
            errs = []
            ap = DATASET / r["audio"]
            if not ap.exists():
                errs.append("missing_audio")
            else:
                try:
                    info = sf.info(str(ap))
                    if info.samplerate != 16000:
                        errs.append(f"sr={info.samplerate}")
                    if info.channels != 1:
                        errs.append(f"ch={info.channels}")
                    if "PCM" not in (info.subtype or ""):
                        errs.append(f"subtype={info.subtype}")
                    filedur = info.frames / info.samplerate
                    if abs(filedur - r["duration"]) > 0.15:
                        errs.append(f"dur_mismatch {filedur:.2f}!={r['duration']}")
                    if filedur > MAX_S:
                        errs.append(f">30s:{filedur:.1f}")
                    if filedur < MIN_S:
                        errs.append(f"<0.5s:{filedur:.2f}")
                except Exception as e:
                    errs.append(f"sf_err:{str(e)[:30]}")
            if not r.get("text", "").strip():
                errs.append("empty_text")
            if r.get("lang") not in ("zh", "ja"):
                errs.append(f"lang={r.get('lang')}")
            ntok = len(tok.encode(r.get("text", ""), add_special_tokens=False))
            if ntok == 0:
                errs.append("zero_tokens")
            if errs:
                fails.append((r.get("audio"), errs))
            else:
                good.append(r)
        cleanp = MANI / f"{split}.clean.jsonl"
        with cleanp.open("w", encoding="utf-8") as f:
            for r in good:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        grand_fail += len(fails)
        print(f"=== {split}: {len(good)}/{len(rows)} pass, {len(fails)} fail -> {cleanp.name}")
        for a, e in fails[:10]:
            print(f"   FAIL {a}: {e}")
    print(f"\n# QA gate: {'ALL PASS' if grand_fail==0 else str(grand_fail)+' failures (已剔除)'}")


if __name__ == "__main__":
    main()
