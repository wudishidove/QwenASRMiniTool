"""build_lexicon.py — DATA-5（bootstrap）：從 pokemon.txt + 語料頻率產 lexicon.tsv 草稿

注意：這是 bootstrap。完整 canonical lexicon 需匯入全寶可夢 zh-TW 名單
（種族/招式/道具/特性，來源神奇寶貝百科 CSV）並人工定案（category/別名）。
本腳本：種子=pokemon.txt 逗號詞；統計 train/test 出現次數；輸出 TSV 供人工擴充。
欄位：canonical  aliases  category  keep_english  freq_train  freq_test
"""
import sys, json, re
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path(__file__).resolve().parents[1] / "dataset"
POKEMON = ROOT / "pokemon.txt"
OUT = DATASET / "lexicon.tsv"


def parse_seed():
    raw = POKEMON.read_text(encoding="utf-8")
    raw = re.sub(r"^[^:：]*[:：]", "", raw, flags=re.M)   # 去 "寶可夢常用詞:" 前綴
    terms = set()
    for tok in re.split(r"[,，\n]", raw):
        t = tok.strip()
        if t and len(t) >= 1:
            terms.add(t)
    return sorted(terms)


def count_freq(terms, manifest):
    rows = [json.loads(l) for l in Path(manifest).open(encoding="utf-8")]
    corpus = "\n".join(r["text"] for r in rows)
    return {t: corpus.count(t) for t in terms}


def main():
    seed = parse_seed()
    ft = count_freq(seed, DATASET / "manifests" / "train.clean.jsonl")
    fe = count_freq(seed, DATASET / "manifests" / "test.clean.jsonl")
    lines = ["canonical\taliases\tcategory\tkeep_english\tfreq_train\tfreq_test"]
    for t in sorted(seed, key=lambda x: -(ft[x] + fe[x])):
        keep_en = "true" if re.fullmatch(r"[A-Za-z0-9 ]+", t) else "false"
        lines.append(f"{t}\t\tunknown\t{keep_en}\t{ft[t]}\t{fe[t]}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"# lexicon bootstrap: {len(seed)} terms -> {OUT}")
    print("# top by freq:")
    for t in sorted(seed, key=lambda x: -(ft[x] + fe[x]))[:15]:
        print(f"   {t:8} train={ft[t]:4} test={fe[t]:3}")
    appear_test = sum(1 for t in seed if fe[t] > 0)
    print(f"# {appear_test}/{len(seed)} 種子詞出現在 test(yt5)")
    print("# 註：完整 entity-F1 需擴充全寶可夢名單（人工/匯入 CSV）")


if __name__ == "__main__":
    main()
