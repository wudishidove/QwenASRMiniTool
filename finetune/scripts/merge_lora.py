"""merge_lora.py — ERROR-5：LoRA merge → 重組完整 top-level HF 目錄 → 載入相容性閘

步驟：
  1. 複製原始 Qwen3-ASR-* 目錄到 out（保留 config/preprocessor/chat_template/
     generation_config/tokenizer/特殊 token）
  2. 載入 base + LoRA adapter → merge_and_unload()（LoRA 併入 thinker 權重）
  3. 刪舊 model*.safetensors / index → merged_model.save_pretrained(out)（只覆寫權重+config）
     tokenizer/preprocessor 等由步驟 1 保留
  4. 載入相容性閘：qwen_asr.Qwen3ASRModel.from_pretrained(out) → 3 段 transcribe()
     → parse_asr_output() 無例外
輸出建議放 GPUModel/pkm-ft-vN/（不覆寫原始）。

用法：
  python merge_lora.py --base <0.6B dir> --adapter adapters/overfit/last \
      --out ../GPUModel/pkm-ft-v0 --gate-manifest dataset/manifests/train.clean.jsonl
"""
import sys, json, argparse, shutil
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(__file__).resolve().parents[1] / "dataset"


def merge(base_dir, adapter_dir, out_dir):
    import torch
    from qwen_asr import Qwen3ASRModel
    from peft import PeftModel

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    print(f"[1] copy base dir -> {out}", flush=True)
    shutil.copytree(base_dir, out, ignore=shutil.ignore_patterns(".cache", "*.md", ".gitattributes"))

    print("[2] load base + adapter, merge_and_unload", flush=True)
    m = Qwen3ASRModel.from_pretrained(base_dir, dtype=torch.bfloat16,
                                      attn_implementation="sdpa", device_map="cpu")
    model = m.model
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()        # 回 top-level Qwen3ASRForConditionalGeneration

    print("[3] remove old weights, save merged", flush=True)
    for f in list(out.glob("model*.safetensors")) + list(out.glob("model.safetensors.index.json")):
        f.unlink()
    # sanitize generation_config（transformers 4.57 嚴格驗證：do_sample=False 不可有 temperature）
    gc = model.generation_config
    gc.do_sample = False
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(gc, attr):
            setattr(gc, attr, None)
    model.save_pretrained(str(out), safe_serialization=True)
    print(f"    merged files: {sorted(p.name for p in out.glob('*.safetensors'))}")
    return out


def load_gate(out_dir, gate_manifest, n=3):
    import torch
    from qwen_asr import Qwen3ASRModel, parse_asr_output
    print("[4] load-compat gate via qwen_asr.Qwen3ASRModel.from_pretrained", flush=True)
    m = Qwen3ASRModel.from_pretrained(str(out_dir), dtype=torch.bfloat16,
                                      attn_implementation="sdpa", device_map="cuda")
    import soundfile as sf
    rows = [json.loads(l) for l in Path(gate_manifest).open(encoding="utf-8")][:n]
    ok = 0
    for r in rows:
        wav, _ = sf.read(str(DATASET / r["audio"]), dtype="float32", always_2d=False)
        res = m.transcribe([(wav, 16000)], language="Chinese")
        txt = res[0].text if res else ""
        # parse_asr_output 不該報錯（raw 解碼字串）
        try:
            parse_asr_output(f"language Chinese<asr_text>{txt}", user_language="Chinese")
        except Exception as e:
            print(f"   parse_asr_output FAIL: {e}")
            continue
        ok += 1
        print(f"   ref: {r['text'][:40]}\n   hyp: {txt[:40]}")
    print(f"[gate] {ok}/{len(rows)} transcribe+parse ok")
    return ok == len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "GPUModel" / "Qwen3-ASR-0.6B"))
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-manifest", default=str(DATASET / "manifests" / "train.clean.jsonl"))
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()

    out = merge(args.base, args.adapter, args.out)
    if not args.skip_gate:
        passed = load_gate(out, args.gate_manifest)
        print(f"\n{'✅ MERGE + LOAD GATE PASSED' if passed else '❌ GATE FAILED'} -> {out}")
    else:
        print(f"\n✅ merged -> {out} (gate skipped)")


if __name__ == "__main__":
    main()
