"""Annotate important lines with the n-gram rarity detector (LLM-annotator-free).

For each solution, score lines by corpus n-gram rarity, keep the top ~32%
(matching the corpus-average important fraction), group contiguous lines into
blocks, and write out the record with `important_lines` replaced — same schema
as the LLM-annotated files, so gen_mutants_lc / unlearn_lc consume it directly.

Usage: python ngram_annotate.py <in.jsonl> <out.jsonl> <code_field>
"""

import json
import math
import os
import sys

import detectors as D

FRAC = 0.32


def annotate(rec, code_field, df, N):
    code = rec[code_field]
    lines = code.split("\n")
    # reuse detector on a shim record
    shim = {"generated_solution": code}
    scores = D.ngram_scores(shim, df, N)
    content = [i for i, ln in enumerate(lines, 1)
               if ln.strip() and not D._SCAFFOLD_RE.match(ln)]
    budget = max(1, round(FRAC * len(content))) if content else 1
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    picked = sorted(ln for ln, sc in ranked if sc > 0)[:0] or \
        sorted([ln for ln, sc in ranked if sc > 0][:budget])
    # group contiguous line numbers into blocks
    blocks = []
    for ln in sorted(picked):
        if blocks and ln == blocks[-1][1] + 1:
            blocks[-1][1] = ln
        else:
            blocks.append([ln, ln])
    rec = dict(rec)
    rec["important_lines"] = [
        {"start_line": a, "end_line": b,
         "code": "\n".join(lines[a - 1:b]), "reason": "ngram-rarity"}
        for a, b in blocks]
    return rec


def main():
    inp, outp, field = sys.argv[1], sys.argv[2], sys.argv[3]
    # idf corpus: all solved leetcode generated solutions (model-specific)
    corpus = []
    for l in open(os.environ.get("LC_NGRAM_CORPUS", "results_leetcode.jsonl")):
        r = json.loads(l)
        if r["passed"]:
            corpus += r["generated_solution"].split("\n")
    df, N = D.build_ngram_idf(corpus)

    n = nb = 0
    with open(outp, "w") as fo:
        for line in open(inp):
            r = json.loads(line)
            if field not in r or not r[field]:
                continue
            a = annotate(r, field, df, N)
            fo.write(json.dumps(a) + "\n")
            n += 1
            nb += len(a["important_lines"])
    print(f"annotated {n} records, {nb} blocks ({nb/n:.2f}/rec) -> {outp}")


if __name__ == "__main__":
    main()
