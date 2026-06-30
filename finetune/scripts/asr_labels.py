"""asr_labels.py — 手建 Qwen3-ASR 訓練 input_ids / labels（ERROR-3 / ERROR-4 修正）

為何手建：processing_qwen3_asr.py 的 __call__ 是 pass-through、chat_template 無
{% generation %} block，沒有 apply_chat_template(output_labels=True)。

做法（與推理 prompt 完全一致 + 附加監督目標）：
  1. 用 processor 產生 prompt+audio 的 input_ids（**動態** audio_pad 數量，
     由音訊長度決定；絕不寫死 390）→ prefix + N×audio_pad + suffix("assistant\n")
  2. 附加 language_suffix(zh) [11528,8453,151704] + 轉錄 token + eos(151645)
  3. labels：prompt 段全 -100，只監督 language_suffix + 轉錄 + eos
loss 走 thinker.forward(labels=...)，HF loss_function 內部自動 shift，
故 labels 與 input_ids 同長、同位置放 target token。
"""
from __future__ import annotations
import json
from pathlib import Path
import torch

DEFAULT_TEMPLATE = Path(__file__).resolve().parents[2] / "prompt_template.json"
IGNORE = -100


def build_messages(context: str = ""):
    return [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]


class LabelBuilder:
    """每段呼叫 build()，回傳單筆 tensor dict（未 padding；padding 交給 collator）。"""

    def __init__(self, processor, template_path=DEFAULT_TEMPLATE, language: str = "Chinese"):
        self.processor = processor
        self.tok = processor.tokenizer
        t = json.loads(Path(template_path).read_text(encoding="utf-8"))
        self.audio_pad_id = int(t["audio_pad_id"])          # 151676
        self.eos_id = int(t["eos_id"])                      # 151645
        if language not in t["language_suffix_ids"]:
            raise ValueError(f"language {language} not in template")
        self.lang_suffix = list(t["language_suffix_ids"][language])  # zh [11528,8453,151704]

    def build(self, wav, text: str, context: str = "") -> dict:
        base = self.processor.apply_chat_template(
            build_messages(context), add_generation_prompt=True, tokenize=False
        )
        enc = self.processor(text=[base], audio=[wav], return_tensors="pt", padding=True)
        prompt_ids = enc["input_ids"][0]                    # prefix + N*audio_pad + suffix
        dtype = prompt_ids.dtype

        tgt = self.tok.encode(text, add_special_tokens=False)
        tail = self.lang_suffix + tgt + [self.eos_id]
        tail_t = torch.tensor(tail, dtype=dtype)

        input_ids = torch.cat([prompt_ids, tail_t])
        labels = torch.cat([torch.full_like(prompt_ids, IGNORE), tail_t.clone()])
        attention_mask = torch.ones_like(input_ids)

        # ── 安全網斷言（ERROR-4）──────────────────────────
        assert labels.shape == input_ids.shape, "labels 長度必須 == input_ids"
        assert int((labels != IGNORE).sum()) == len(tail), "監督 token 數不符"
        n_pad = int((input_ids == self.audio_pad_id).sum())
        assert n_pad > 0, "找不到 audio_pad，prompt/audio 構造錯誤"

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": enc["input_features"][0],
            "feature_attention_mask": enc["feature_attention_mask"][0],
            "n_audio_pad": n_pad,
            "n_supervised": len(tail),
        }


def collate(batch: list[dict], pad_token_id: int = 151643) -> dict:
    """batch padding：input_ids/attention_mask 右 pad，labels 右 pad 填 -100，
    input_features/feature_attention_mask 右 pad（沿 frame 軸）。"""
    import torch.nn.functional as F
    maxlen = max(b["input_ids"].size(0) for b in batch)
    maxframe = max(b["input_features"].size(-1) for b in batch)

    input_ids, attn, labels, feats, fmask = [], [], [], [], []
    for b in batch:
        L = b["input_ids"].size(0)
        padL = maxlen - L
        input_ids.append(F.pad(b["input_ids"], (0, padL), value=pad_token_id))
        attn.append(F.pad(b["attention_mask"], (0, padL), value=0))
        labels.append(F.pad(b["labels"], (0, padL), value=IGNORE))
        Fr = b["input_features"].size(-1)
        padF = maxframe - Fr
        feats.append(F.pad(b["input_features"], (0, padF), value=0.0))
        fmask.append(F.pad(b["feature_attention_mask"], (0, padF), value=0))

    out = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attn),
        "labels": torch.stack(labels),
        "input_features": torch.stack(feats),
        "feature_attention_mask": torch.stack(fmask),
    }
    # 斷言：每筆 labels 長度 == input_ids 長度
    assert out["labels"].shape == out["input_ids"].shape
    return out
