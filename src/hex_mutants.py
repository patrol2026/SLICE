"""Build verified wrong-core mutants + important-line blocks for HumanEval-X
forget (and guard) solutions, per language, per detector.

Output: hex_annot_<detector>.json
  {lang: {task_num: {"code":..., "prompt":..., "test":...,
                     "blocks":[{start,end,orig,mutated}...]}}}
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import detectors_x as DX
import execx
from gen_mutants import try_mutations

FRAC = 0.34


def blocks_from_scores(code, scores):
    lines = code.split("\n")
    content = [i for i, ln in enumerate(lines, 1) if ln.strip()]
    budget = max(1, round(FRAC * len(content)))
    picked = sorted([ln for ln, sc in
                     sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                     if sc > 0][:budget])
    blocks = []
    for ln in picked:
        if blocks and ln == blocks[-1][1] + 1:
            blocks[-1][1] = ln
        else:
            blocks.append([ln, ln])
    return blocks


def verified_mutant(lang, code, prompt, test, a, b, pool):
    """Return a mutation of lines a..b that compiles+fails tests, else None."""
    lines = code.split("\n")
    orig = "\n".join(lines[a - 1:b])
    for cand in list(try_mutations(orig))[:6]:
        mut_code = "\n".join(lines[:a - 1] + cand.split("\n") + lines[b:])
        src = execx.build_source(lang, prompt, mut_code, test)
        passed, _ = execx.run_source(lang, src)
        if not passed:                       # mutation broke behavior -> valid
            return {"start": a, "end": b, "orig": orig, "mutated": cand}
    return None


def main():
    detector = sys.argv[1]   # ast | dataflow | tfidf
    score_fn = {"ast": DX.ast_scores, "dataflow": DX.dataflow_scores,
                "tfidf": DX.tfidf_scores}[detector]
    splits = json.load(open("hex_splits.json"))
    want = set(splits["forget"]) | set(splits["retain"])
    pool = ThreadPoolExecutor(max_workers=16)

    out = {}
    for lang in ["python", "java", "cpp"]:
        recs = {j["task_id"].split("/")[-1]: j
                for j in map(json.loads, open(f"results_hex_{lang}.jsonl"))
                if j["passed"]}
        out[lang] = {}
        n_blk = 0
        for num in sorted(want, key=int):
            if num not in recs:
                continue
            r = recs[num]
            code = r["generated_code"]
            blocks = blocks_from_scores(code, score_fn(code, lang))
            muts = []
            for a, b in blocks:
                m = verified_mutant(lang, code, r["prompt"], r["test"], a, b, pool)
                if m:
                    muts.append(m)
            if muts:
                out[lang][num] = {"code": code, "prompt": r["prompt"],
                                  "test": r["test"], "blocks": muts}
                n_blk += len(muts)
        print(f"{detector}/{lang}: {len(out[lang])} tasks, {n_blk} verified mutant blocks",
              flush=True)
    json.dump(out, open(f"hex_annot_{detector}.json", "w"))
    print(f"saved hex_annot_{detector}.json")


if __name__ == "__main__":
    main()
