# Repository Guidelines

## Project Structure & Module Organization

This directory is the Qwen3-ASR Pokemon battle ASR fine-tuning workspace. Core pipeline code lives in `scripts/`, with numbered scripts for data preparation (`01_download.py` through `04_qa_gate.py`), training (`train_lora.py`), merging (`merge_lora.py`), evaluation (`eval.py`), and validation helpers (`verify_harness.py`). Dataset inputs, manifests, lexicons, and generated metadata live under `dataset/`, especially `dataset/manifests/`. LoRA checkpoints are stored in `adapters/`, and evaluation outputs are stored in `reports/`. Long-form decisions and experimental results are documented in `結果.md`; `CLAUDE.md` is the high-level architecture index.

## Build, Test, and Development Commands

Use the repository training environment for all scripts:

```powershell
$env:PYTHONIOENCODING='utf-8'
$PY='venv-train/Scripts/python.exe'
```

Common commands:

```powershell
& $PY scripts/verify_harness.py
& $PY scripts/03_make_clips.py
& $PY scripts/04_qa_gate.py
& $PY scripts/train_lora.py --model ../GPUModel/Qwen3-ASR-1.7B --train dataset/manifests/train.clean.jsonl --val dataset/manifests/val.clean.jsonl --epochs 3 --batch 4 --grad-accum 8 --lr 2e-4 --r 16 --alpha 32 --eval-every 40 --out adapters/p3-1.7b-v2
& $PY scripts/merge_lora.py --base ../GPUModel/Qwen3-ASR-1.7B --adapter adapters/p3-1.7b-v2/best --out ../GPUModel/pkm-ft-1.7b-v2 --skip-gate
& $PY scripts/eval.py --model ../GPUModel/pkm-ft-1.7b-v2 --context off --out reports/B2_1.7b_off.json
```

## Coding Style & Naming Conventions

Code is Python 3.12-oriented and uses 4-space indentation. Keep scripts CLI-friendly with `argparse`, `main()`, and `if __name__ == "__main__":`. Prefer `pathlib.Path` and explicit UTF-8 file I/O, for example `Path(path).open(encoding="utf-8")`. In PowerShell file reads/writes, include `-Encoding UTF8`. Preserve existing manifest names such as `train.clean.jsonl`, `val.v4.jsonl`, and report names like `B2_1.7b_v4_last_off.json`.

## Testing Guidelines

There is no separate pytest suite. Use `scripts/verify_harness.py` as the wiring smoke test before changing label construction, LoRA targeting, collators, or model loading. For training or evaluation changes, run a limited job first with `--limit` or `--minutes`, then run `eval.py` against the relevant manifest and save JSON under `reports/`.

## Commit & Pull Request Guidelines

Git history uses Conventional Commit prefixes such as `feat:`, `fix:`, `fix(security):`, and `build:`; continue that pattern with concise Chinese or English summaries. PRs should describe the affected pipeline stage, list commands run, include key CER/entity-F1 changes when relevant, and mention any new or changed files in `dataset/`, `adapters/`, or `reports/`.

## Agent-Specific Notes

Do not upgrade `transformers` without explicit approval. Do not apply OpenCC to training targets or model outputs. Keep deployment assumptions aligned with `CLAUDE.md`: v2 merged model, context off, OpenCC off.
