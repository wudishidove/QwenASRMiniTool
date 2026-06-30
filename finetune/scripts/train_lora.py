"""train_lora.py — Phase 1/2 LoRA 訓練（手寫迴圈，已驗證接線）

實作要點（全部已於 verify_harness.py 對真實 0.6B 驗證）：
  [BLOCKER-1] forward 走 model.thinker(**batch).loss
  [BLOCKER-2] LoRA 精確 regex 只掛 thinker LM（audio_tower 0 命中）
  [ERROR-3/4] 手建 labels（asr_labels.LabelBuilder）動態 audio_pad、只監督尾巴
  [GOTCHA] grad-ckpt / input-require-grads 對 model.thinker 呼叫

用法（煙霧過擬合）：
  python train_lora.py --train manifests/train.clean.jsonl --limit 30 \
      --epochs 50 --batch 2 --lr 2e-4 --out adapters/overfit --no-replay
用法（Phase 1 煙霧）：
  python train_lora.py --train ... --val ... --minutes 25 --epochs 3 --out adapters/p1
"""
import sys, json, argparse, random, math, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import Dataset, DataLoader
from asr_labels import LabelBuilder, collate

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(__file__).resolve().parents[1] / "dataset"
DEFAULT_MODEL = str(ROOT / "GPUModel" / "Qwen3-ASR-0.6B")
LORA_REGEX = r"thinker\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


class ClipDataset(Dataset):
    def __init__(self, rows, builder):
        self.rows = rows
        self.lb = builder

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import soundfile as sf
        r = self.rows[i]
        wav, _ = sf.read(str(DATASET / r["audio"]), dtype="float32", always_2d=False)
        lang = "Japanese" if r.get("lang") == "ja" else "Chinese"
        lb = self.lb if lang == "Chinese" else LabelBuilder(self.lb.processor, language=lang)
        return lb.build(wav, r["text"], context=r.get("context", ""))


def load_rows(path, minutes=None, limit=None, seed=0):
    rows = [json.loads(l) for l in Path(path).open(encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    if minutes:
        out, acc = [], 0.0
        for r in rows:
            out.append(r); acc += r["duration"]
            if acc >= minutes * 60:
                break
        rows = out
    if limit:
        rows = rows[:limit]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--replay", default=None, help="通用 replay manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=0, help="每 N optimizer step 算 val loss")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from qwen_asr import Qwen3ASRModel
    from peft import LoraConfig, get_peft_model

    print(f"== load {Path(args.model).name} ==", flush=True)
    m = Qwen3ASRModel.from_pretrained(args.model, dtype=torch.bfloat16,
                                      attn_implementation="sdpa", device_map="cuda")
    model, proc = m.model, m.processor
    dev = model.device

    for p in model.parameters():
        p.requires_grad = False
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
        task_type="CAUSAL_LM", target_modules=LORA_REGEX))
    model.print_trainable_parameters()
    base = model.base_model.model            # 原 top-level（LoRA 已注入 submodule）
    base.thinker.gradient_checkpointing_enable()
    base.thinker.enable_input_require_grads()

    lb = LabelBuilder(proc, language="Chinese")
    rows = load_rows(args.train, args.minutes, args.limit, args.seed)
    if args.replay:
        rep = load_rows(args.replay, minutes=(args.minutes if args.minutes else None))
        rows = rows + rep
        random.Random(args.seed).shuffle(rows)
    print(f"train rows: {len(rows)} ({sum(r['duration'] for r in rows)/60:.1f}min)", flush=True)
    dl = DataLoader(ClipDataset(rows, lb), batch_size=args.batch, shuffle=True,
                    num_workers=0, collate_fn=collate)
    val_rows = load_rows(args.val, limit=200) if args.val else None

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(dl) / args.grad_accum)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup))

    def lr_at(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    @torch.no_grad()
    def val_loss():
        if not val_rows:
            return None
        model.eval()
        vlb = LabelBuilder(proc, language="Chinese")
        import soundfile as sf
        tot, n = 0.0, 0
        for r in val_rows[:100]:
            wav, _ = sf.read(str(DATASET / r["audio"]), dtype="float32", always_2d=False)
            s = vlb.build(wav, r["text"])
            b = collate([s])
            b = {k: (v.to(dev).to(base.dtype) if v.dtype.is_floating_point else v.to(dev)) for k, v in b.items()}
            tot += float(base.thinker(**b).loss); n += 1
        model.train()
        return tot / max(1, n)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.train()
    gstep = 0
    best_val = float("inf")
    t0 = time.time()
    for ep in range(args.epochs):
        opt.zero_grad()
        running = 0.0
        for i, batch in enumerate(dl):
            batch = {k: (v.to(dev).to(base.dtype) if v.dtype.is_floating_point else v.to(dev))
                     for k, v in batch.items()}
            loss = base.thinker(**batch).loss / args.grad_accum
            loss.backward()
            running += float(loss) * args.grad_accum
            if (i + 1) % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = args.lr * lr_at(gstep)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(); gstep += 1
                if gstep % 5 == 0:
                    avg = running / ((i + 1))
                    print(f"ep{ep} step{gstep}/{total_steps} loss {avg:.4f} lr {opt.param_groups[0]['lr']:.2e} {(time.time()-t0):.0f}s", flush=True)
                if args.eval_every and gstep % args.eval_every == 0:
                    vl = val_loss()
                    print(f"  [val] loss {vl:.4f}", flush=True)
                    if vl is not None and vl < best_val:
                        best_val = vl
                        model.save_pretrained(str(Path(args.out) / "best"))
                if args.max_steps and gstep >= args.max_steps:
                    break
        epoch_loss = running / max(1, len(dl))
        vl = val_loss()
        print(f"== epoch {ep} done: train_loss {epoch_loss:.4f}" + (f" val_loss {vl:.4f}" if vl else "") + " ==", flush=True)
        model.save_pretrained(str(Path(args.out) / f"epoch{ep}"))
        if args.max_steps and gstep >= args.max_steps:
            break

    model.save_pretrained(str(Path(args.out) / "last"))
    print(f"\n✅ saved adapters -> {args.out}  (best_val={best_val if best_val<1e9 else 'NA'})")


if __name__ == "__main__":
    main()
