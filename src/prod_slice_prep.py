"""Prepare the copyrighted-code SLICE run from the memorization census.

1. Splits: memorized files (BLEU >= 0.3) -> forget/heldout/retain 20/20/60,
   seed 42.  -> prod_splits.json
2. Important lines: n-gram rarity over the 1,100-file corpus (budget 34%),
   contiguous lines merged into blocks.
3. Mutants: try_mutations per block, keep first candidate where the FULL
   mutated file still parses (ast.parse) - the no-test-oracle analog of the
   LeetCode verified mutant.  -> prod_slice_data.json
"""

import ast as pyast
import json
import random

from datasets import Dataset

import detectors as D
from gen_mutants import try_mutations

THR = 0.3
FRAC = 0.34
SEED = 42
import os
RATIO_F = float(os.environ.get("PROD_RATIO_F", 0.2))
RATIO_H = float(os.environ.get("PROD_RATIO_H", 0.2))
SPLITS_OUT = os.environ.get("PROD_SPLITS_OUT", "prod_splits.json")
DATA_OUT = os.environ.get("PROD_DATA_OUT", "prod_slice_data.json")


def load_corpus():
    files = {}
    forget = Dataset.from_file("PROD/data/forget_data/data-00000-of-00001.arrow")
    for r in forget:
        files[f"forget:{r['task_id']}"] = r["canonical_solution"]
    for l in open("PROD/data/starcoder_pool_1000.jsonl"):
        r = json.loads(l)
        files[f"pool:{r['pool_id']}"] = r["code"]
    return files


def blocks_from_scores(code, scores):
    lines = code.split("\n")
    content = [i for i, ln in enumerate(lines, 1) if ln.strip()]
    budget = max(1, round(FRAC * len(content)))
    picked = sorted([ln for ln, sc in
                     sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                     if sc > 0][:budget])
    out = []
    for ln in picked:
        if out and ln == out[-1][1] + 1:
            out[-1][1] = ln
        else:
            out.append([ln, ln])
    return out


import re as _re


def fallback_mutations(orig):
    """String/identifier/number perturbations for operator-free lines."""
    out = []
    m = _re.search(r"\b\d+\b", orig)
    if m:
        out.append(orig[:m.start()] + str(int(m.group()) + 1) + orig[m.end():])
    m = _re.search(r"(['\"])((?:(?!\1).){3,})\1", orig)
    if m:
        s = m.group(2)
        out.append(orig.replace(m.group(0), m.group(1) + s[::-1] + m.group(1), 1))
    ids = sorted(set(_re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b", orig)),
                 key=len, reverse=True)
    for name in ids[:2]:
        if name in ("self", "None", "True", "False", "return", "import"):
            continue
        out.append(_re.sub(r"\b%s\b" % _re.escape(name), name + "_alt", orig, count=1))
    return out


def verified_mutant(code, a, b, need_parse=True):
    lines = code.split("\n")
    orig = "\n".join(lines[a - 1:b])
    cands = list(try_mutations(orig))[:6] + fallback_mutations(orig)
    for cand in cands:
        if cand == orig:
            continue
        if need_parse:
            mut = "\n".join(lines[:a - 1] + cand.split("\n") + lines[b:])
            try:
                pyast.parse(mut)
            except SyntaxError:
                continue
        return {"start": a, "end": b, "orig": orig, "mutated": cand}
    return None


def main():
    files = load_corpus()
    mem = {}
    for l in open(os.environ.get("PROD_MEMCENSUS","prod_memorization.jsonl")):
        r = json.loads(l)
        mem[f"{r['split_src']}:{r['fid']}"] = r["bleu"]
    qualified = sorted(k for k, v in mem.items() if v >= THR and k in files)
    print(f"memorized (BLEU>={THR}): {len(qualified)}/{len(mem)}  "
          f"(forget-origin: {sum(1 for k in qualified if k.startswith('forget'))})")

    rng = random.Random(SEED)
    rng.shuffle(qualified)
    n = len(qualified)
    nf, nh = round(RATIO_F * n), round(RATIO_H * n)
    splits = {"forget": sorted(qualified[:nf]),
              "heldout": sorted(qualified[nf:nf + nh]),
              "retain": sorted(qualified[nf + nh:])}
    json.dump(splits, open(SPLITS_OUT, "w"), indent=1)
    print({k: len(v) for k, v in splits.items()})

    # n-gram IDF over the full 1,100-file corpus
    corpus_lines = []
    for code in files.values():
        corpus_lines += code.split("\n")
    df, N = D.build_ngram_idf(corpus_lines)

    out = {}
    stats = {"files": 0, "blocks": 0, "first_half_blocks": 0, "zero_block": 0}
    stats["py2_files"] = 0
    for k in splits["forget"] + splits["retain"]:
        code = files[k]
        try:
            pyast.parse(code); parses = True
        except SyntaxError:
            parses = False
            stats["py2_files"] += 1
        scores = D.ngram_scores({"generated_solution": code}, df, N)
        blocks = blocks_from_scores(code, scores)
        muts = []
        for a, b in blocks:
            m = verified_mutant(code, a, b, need_parse=parses)
            if m:
                muts.append(m)
        half = len(code.split("\n")) // 2
        stats["files"] += 1
        stats["blocks"] += len(muts)
        stats["first_half_blocks"] += sum(1 for m in muts if m["start"] <= half)
        if not muts:
            stats["zero_block"] += 1
        out[k] = {"code": code, "blocks": muts}
    json.dump(out, open(DATA_OUT, "w"))
    print("detection/mutants:", stats,
          f"(mean blocks/file {stats['blocks']/max(1,stats['files']):.1f})")


if __name__ == "__main__":
    main()
