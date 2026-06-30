"""eval.py — EVAL-0：CER/WER + entity-F1 評測（ref/hyp 共用 text_norm.for_cer）

三 baseline 對應：
  B0 = base 偏置 OFF（--context off）
  B1 = base 偏置 ON（--context bias，餵 pokemon.txt）= 真 floor
  B2 = base+LoRA 偏置 ON（--model <merged> --context bias）
指標：整體 CER（jiwer 字元級）、entity-recall/precision/F1（對 lexicon，macro+micro）。
解碼後 result.text 已是純文字（qwen_asr 已剝 language<asr_text>）。

用法：
  python eval.py --model <dir> --manifest dataset/manifests/test.clean.jsonl \
     --lexicon dataset/lexicon.tsv --context off --out reports/B0.json [--limit 50]
"""
import sys, json, argparse, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_norm import for_cer

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(__file__).resolve().parents[1] / "dataset"
POKEMON = ROOT / "pokemon.txt"


def load_lexicon(path):
    terms = []
    for i, ln in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if i == 0 or not ln.strip():
            continue
        canonical = ln.split("\t")[0].strip()
        if canonical:
            terms.append(for_cer(canonical))
    return [t for t in terms if t]


def bias_context():
    import re
    raw = POKEMON.read_text(encoding="utf-8")
    raw = re.sub(r"^[^:：]*[:：]", "", raw, flags=re.M)
    toks = [t.strip() for t in re.split(r"[,，\n]", raw) if t.strip()]
    return " ".join(toks)


def entity_stats(refs_cer, hyps_cer, lex):
    # per-term TP/FP/FN across all clips
    tp = {t: 0 for t in lex}; fp = {t: 0 for t in lex}; fn = {t: 0 for t in lex}
    for rc, hc in zip(refs_cer, hyps_cer):
        for t in lex:
            in_r = t in rc
            in_h = t in hc
            if in_r and in_h: tp[t] += 1
            elif in_h and not in_r: fp[t] += 1
            elif in_r and not in_h: fn[t] += 1
    # micro
    TP = sum(tp.values()); FP = sum(fp.values()); FN = sum(fn.values())
    micro_p = TP / (TP + FP) if TP + FP else 0.0
    micro_r = TP / (TP + FN) if TP + FN else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    # macro over terms that appear in ref
    fs, rs = [], []
    for t in lex:
        if tp[t] + fn[t] == 0:
            continue
        p = tp[t] / (tp[t] + fp[t]) if tp[t] + fp[t] else 0.0
        r = tp[t] / (tp[t] + fn[t])
        f = 2 * p * r / (p + r) if p + r else 0.0
        fs.append(f); rs.append(r)
    macro_f = sum(fs) / len(fs) if fs else 0.0
    macro_r = sum(rs) / len(rs) if rs else 0.0
    return {"micro_p": micro_p, "micro_r": micro_r, "micro_f": micro_f,
            "macro_f": macro_f, "macro_r": macro_r,
            "n_terms_in_ref": len(fs), "TP": TP, "FP": FP, "FN": FN}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default=str(DATASET / "manifests" / "test.clean.jsonl"))
    ap.add_argument("--lexicon", default=str(DATASET / "lexicon.tsv"))
    ap.add_argument("--context", choices=["off", "bias"], default="off")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch, jiwer, soundfile as sf
    from qwen_asr import Qwen3ASRModel

    ctx = "" if args.context == "off" else bias_context()
    print(f"== eval model={Path(args.model).name} context={args.context} ==", flush=True)
    m = Qwen3ASRModel.from_pretrained(args.model, dtype=torch.bfloat16,
                                      attn_implementation="sdpa", device_map="cuda")
    rows = [json.loads(l) for l in Path(args.manifest).open(encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    lex = load_lexicon(args.lexicon)

    refs, hyps = [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        wav, _ = sf.read(str(DATASET / r["audio"]), dtype="float32", always_2d=False)
        res = m.transcribe([(wav, 16000)], context=ctx, language="Chinese")
        hyp = res[0].text if res else ""
        refs.append(r["text"]); hyps.append(hyp)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    refs_cer = [for_cer(x) for x in refs]
    hyps_cer = [for_cer(x) for x in hyps]
    # 過濾空 ref（CER 無意義）
    pairs = [(a, b) for a, b in zip(refs_cer, hyps_cer) if a]
    cer = jiwer.cer([a for a, b in pairs], [b for a, b in pairs])
    ent = entity_stats(refs_cer, hyps_cer, lex)

    report = {"model": str(args.model), "context": args.context, "n": len(rows),
              "CER": round(cer, 4), "entity": ent,
              "samples": [{"ref": refs[i][:60], "hyp": hyps[i][:60]} for i in range(min(5, len(rows)))]}
    print(f"\nCER = {cer:.4f}  ({len(pairs)} clips)")
    print(f"entity micro_F={ent['micro_f']:.3f} recall={ent['micro_r']:.3f}  "
          f"macro_F={ent['macro_f']:.3f} recall={ent['macro_r']:.3f}  "
          f"(TP{ent['TP']}/FP{ent['FP']}/FN{ent['FN']}, {ent['n_terms_in_ref']} terms)")
    for s in report["samples"]:
        print(f"  ref: {s['ref']}\n  hyp: {s['hyp']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
