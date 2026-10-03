"""Build the CyberSecEval unlearning data for both split protocols.

Pool = snippets the memorized model emits the vulnerability for (funnel) AND that
have a verified secure mutant (cyber_mutants). Split into forget/heldout/retain
(A=10/20/70, B=20/20/60, seed 42) and emit prod_unlearn-format cil data:
  {cid: {code, rule, blocks:[{start,end,orig,mutated}]}}
Outputs cyber_splits_{A,B}.json and cyber_cil_data_{A,B}.json.
"""
import json
import random

SEED = 42
import os
DT = os.environ.get("CY_TAG", "")   # data tag per model, e.g. "_ds"
CENSUS = os.environ.get("CY_MEMCENSUS", "cyber_memorization.jsonl")


def main():
    recs = {r["cid"]: r for r in map(json.loads, open("cyber_records.jsonl"))}
    muts = {m["cid"]: m["block"] for m in map(json.loads, open("cyber_mutants.jsonl"))}
    census = {r["cid"]: r for r in map(json.loads, open(CENSUS))}

    pool = sorted(cid for cid in recs
                  if cid in muts and census.get(cid, {}).get("vuln_emitted"))
    print(f"pool (memorized-emits-vuln AND has verified mutant): {len(pool)}")

    data = {cid: {"code": recs[cid]["code"], "rule": recs[cid]["rule"],
                  "blocks": [muts[cid]]} for cid in pool}

    for tag, (rf, rh) in {"A": (0.10, 0.20), "B": (0.20, 0.20)}.items():
        rng = random.Random(SEED)
        order = pool[:]
        rng.shuffle(order)
        n = len(order)
        nf, nh = round(rf * n), round(rh * n)
        splits = {"forget": sorted(order[:nf]),
                  "heldout": sorted(order[nf:nf + nh]),
                  "retain": sorted(order[nf + nh:])}
        json.dump(splits, open(f"cyber_splits{DT}_{tag}.json", "w"), indent=1)
        json.dump(data, open(f"cyber_cil_data{DT}_{tag}.json", "w"))
        print(f"  split {tag}: " + " ".join(f"{k}={len(v)}" for k, v in splits.items()))


if __name__ == "__main__":
    main()
