"""05_forgetting_check.py — 快速遺忘抽查（通用中文）

抓少量 out-of-domain 通用中文（FLEURS cmn / 讀稿語音，串流免完整下載），
比較 base 1.7B vs 微調 1.7B 的 CER。微調若大幅退化＝災難性遺忘。
（FLEURS 為簡體讀稿；for_cer 對 ref/hyp 對稱 t2s，故繁簡無關。）

用法： python 05_forgetting_check.py --n 40
"""
import sys, argparse, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_norm import for_cer

ROOT = Path(__file__).resolve().parents[2]
BASE = str(ROOT / "GPUModel" / "Qwen3-ASR-1.7B")
FT = str(ROOT / "GPUModel" / "pkm-ft-1.7b-v1")


def get_samples(n):
    from datasets import load_dataset
    import numpy as np
    ds = load_dataset("google/fleurs", "cmn_hans_cn", split="test",
                      streaming=True, trust_remote_code=True)
    out = []
    for ex in ds:
        a = ex["audio"]
        wav = np.asarray(a["array"], dtype="float32")
        sr = a["sampling_rate"]
        if sr != 16000:
            import soxr
            wav = soxr.resample(wav, sr, 16000).astype("float32")
        out.append((wav, ex.get("transcription") or ex.get("raw_transcription") or ""))
        if len(out) >= n:
            break
    return out


def eval_model(path, samples):
    import torch, jiwer
    from qwen_asr import Qwen3ASRModel
    m = Qwen3ASRModel.from_pretrained(path, dtype=torch.bfloat16,
                                      attn_implementation="sdpa", device_map="cuda")
    refs, hyps = [], []
    for wav, ref in samples:
        res = m.transcribe([(wav, 16000)], context="", language="Chinese")
        hyps.append(res[0].text if res else "")
        refs.append(ref)
    del m
    torch.cuda.empty_cache()
    pairs = [(for_cer(a), for_cer(b)) for a, b in zip(refs, hyps) if for_cer(a)]
    cer = jiwer.cer([a for a, b in pairs], [b for a, b in pairs])
    return cer, list(zip(refs[:3], hyps[:3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    print(f"== fetch {args.n} general-zh (FLEURS cmn) samples ==", flush=True)
    samples = get_samples(args.n)
    print(f"got {len(samples)} samples", flush=True)

    print("== base 1.7B ==", flush=True)
    base_cer, base_s = eval_model(BASE, samples)
    print(f"base general CER = {base_cer:.4f}", flush=True)

    print("== finetuned 1.7B (no replay) ==", flush=True)
    ft_cer, ft_s = eval_model(FT, samples)
    print(f"finetuned general CER = {ft_cer:.4f}", flush=True)

    delta = ft_cer - base_cer
    print(f"\n=== FORGETTING: base {base_cer:.4f} -> ft {ft_cer:.4f}  (Δ {delta:+.4f}) ===")
    print("verdict:", "OK(輕微)" if delta <= 0.03 else ("中度退化" if delta <= 0.08 else "明顯遺忘→需 replay"))
    for r, h in ft_s:
        print(f"  ref: {r[:50]}\n  hyp: {h[:50]}")


if __name__ == "__main__":
    main()
