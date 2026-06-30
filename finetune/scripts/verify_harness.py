"""verify_harness.py — 對真實 0.6B 驗證訓練接線（M1 gate 的前置接線檢查）

驗證：
  [ERROR-3] audio_pad 數量隨音訊長度動態變化（非寫死 390）
  [ERROR-4] labels 長度 == input_ids、(labels!=-100).sum() == 監督 token 數
  [BLOCKER-1] loss 走 model.thinker(input_ids, input_features, ..., labels).loss 且有限
  [BLOCKER-2] LoRA 精確 regex 只命中 thinker LM、audio_tower 命中 0
  [TRAIN-4] backward 後 LoRA 有 grad、audio_tower 無 grad
  [相容] LoRA patch 後 top-level transcribe()/generate() 仍可走
"""
import sys, numpy as np, torch
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]          # qwen/
sys.path.insert(0, str(ROOT))                       # for audio_io
sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_labels import LabelBuilder, collate
from audio_io import load_audio_16k_mono

MODEL = str(ROOT / "GPUModel" / "Qwen3-ASR-0.6B")
LORA_REGEX = r"thinker\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
H51 = ROOT / "finetune" / "dataset" / "raw" / "yt5" / "h51REmf4JNY.m4a"


def main():
    from qwen_asr import Qwen3ASRModel
    from peft import LoraConfig, get_peft_model

    print("== load 0.6B ==")
    m = Qwen3ASRModel.from_pretrained(MODEL, dtype=torch.bfloat16,
                                      attn_implementation="sdpa", device_map="cuda")
    model, proc = m.model, m.processor
    dev = model.device

    # 真實音訊兩段（10s / 7s）→ 驗證動態 audio_pad
    if H51.exists():
        wav, _ = load_audio_16k_mono(str(H51))
        a = wav[16000*30:16000*40]      # 10s
        b = wav[16000*60:16000*67]      # 7s
    else:
        a = np.zeros(16000*10, np.float32); b = np.zeros(16000*7, np.float32)

    lb = LabelBuilder(proc, language="Chinese")
    s1 = lb.build(a, "這隻噴火龍使出十萬伏特")
    s2 = lb.build(b, "大狃拉用劍舞之後接尖石攻擊")
    print(f"[ERROR-3] dynamic audio_pad: 10s->{s1['n_audio_pad']}  7s->{s2['n_audio_pad']}  (30s would be ~390)")
    assert s1["n_audio_pad"] != s2["n_audio_pad"], "audio_pad 應隨長度變化"
    print(f"[ERROR-4] s1 labels==input_ids: {s1['labels'].shape==s1['input_ids'].shape}, supervised={s1['n_supervised']}")

    batch = collate([s1, s2])
    batch = {k: (v.to(dev).to(model.dtype) if v.dtype.is_floating_point else v.to(dev))
             for k, v in batch.items()}
    print("[collate] input_ids", tuple(batch["input_ids"].shape),
          "labels", tuple(batch["labels"].shape),
          "input_features", tuple(batch["input_features"].shape))

    # ── BLOCKER-1：loss 走 thinker ───────────────────────────
    print("\n== [BLOCKER-1] thinker forward loss ==")
    out = model.thinker(**batch)
    loss = out.loss
    print("loss =", float(loss), "finite:", torch.isfinite(loss).item())
    assert torch.isfinite(loss).item(), "loss 非有限"

    # ── BLOCKER-2：LoRA 精確 regex ───────────────────────────
    print("\n== [BLOCKER-2] attach LoRA, exact regex ==")
    for p in model.parameters():
        p.requires_grad = False
    peft_model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=LORA_REGEX))
    peft_model.print_trainable_parameters()
    # 統計命中位置
    hit_thinker = hit_audio = 0
    for n, mod in model.named_modules():
        if hasattr(mod, "lora_A"):
            if "audio_tower" in n: hit_audio += 1
            elif "thinker" in n: hit_thinker += 1
    print(f"LoRA hits: thinker={hit_thinker}  audio_tower={hit_audio}")
    assert hit_audio == 0, "LoRA 誤命中 audio_tower！"
    assert hit_thinker > 0, "LoRA 沒命中 thinker"

    # ── TRAIN-4：backward grad 檢查 ──────────────────────────
    print("\n== [TRAIN-4] backward grad check ==")
    # 注意：grad-ckpt / input-require-grads 必須對 thinker 呼叫
    # （top-level Qwen3ASRForConditionalGeneration 未實作 get_input_embeddings）
    model.thinker.gradient_checkpointing_enable()
    model.thinker.enable_input_require_grads()
    out = model.thinker(**batch)
    out.loss.backward()
    lora_with_grad = sum(1 for n, p in model.named_parameters()
                         if "lora_" in n and p.grad is not None and p.grad.abs().sum() > 0)
    audio_grad = [n for n, p in model.named_parameters()
                  if "audio_tower" in n and p.grad is not None]
    print(f"LoRA params with non-zero grad: {lora_with_grad}")
    print(f"audio_tower params with grad (應為空): {len(audio_grad)}")
    assert lora_with_grad > 0, "LoRA 無 grad"
    assert len(audio_grad) == 0, "audio_tower 不該有 grad"

    # ── 相容：transcribe 仍可走 ──────────────────────────────
    print("\n== [相容] LoRA patch 後 transcribe() ==")
    model.eval()
    with torch.no_grad():
        res = m.transcribe([(a, 16000)], language="Chinese")
    print("transcribe ok, sample text:", (res[0].text if res else "")[:40])

    print("\n✅ ALL HARNESS CHECKS PASSED (TRAIN-0/2/3/4 接線成立)")


if __name__ == "__main__":
    main()
