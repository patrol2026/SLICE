"""Stage the DeepSeek run after its sweep: splits, baseline summary, annotation
chunks for the forget set's generated solutions."""

import json
import os
import random

RESULTS = "results_leetcode_dsv2.jsonl"

res = {json.loads(l)["task_id"]: json.loads(l) for l in open(RESULTS)}
ds = {json.loads(l)["task_id"]: json.loads(l)
      for l in open("leetcode_important_lines.jsonl")}
pool = sorted(t for t in ds
              if res.get(t, {}).get("passed") and ds[t]["important_lines"])
print(f"solved+annotated pool: {len(pool)}")

rng = random.Random(42)
rng.shuffle(pool)
n = len(pool)
nf, nh = round(n * 0.10), round(n * 0.20)
splits = {"forget": sorted(pool[:nf]), "heldout": sorted(pool[nf:nf + nh]),
          "retain": sorted(pool[nf + nh:])}
json.dump(splits, open("leetcode_splits_ds.json", "w"), indent=1)
print({k: len(v) for k, v in splits.items()})

# Baseline summary for the ds split, free from sweep results.
summary = {}
for s in ("forget", "heldout", "retain"):
    rs = [res[t] for t in splits[s]]
    summary[s] = {"passed": sum(r["passed"] for r in rs), "total": len(rs),
                  "pass@1": round(sum(r["passed"] for r in rs) / len(rs), 4)}
all_s = json.load(open("lc_summary.json"))
all_s["baseline_ds"] = summary
json.dump(all_s, open("lc_summary.json", "w"), indent=1)
print("baseline_ds:", {k: v["pass@1"] for k, v in summary.items()})

# Annotation chunks of the forget set's GENERATED solutions.
outdir = "./scratch/ds_annot"
os.makedirs(outdir, exist_ok=True)
CH = 50
forget = splits["forget"]
for ci in range(0, len(forget), CH):
    with open(f"{outdir}/chunk_{ci//CH}.txt", "w") as f:
        for tid in forget[ci:ci + CH]:
            f.write(f"### {tid}\n")
            for i, line in enumerate(res[tid]["generated_solution"].split("\n"), 1):
                f.write(f"{i:3d}| {line}\n")
            f.write("\n")
print(f"{(len(forget)+CH-1)//CH} annotation chunks written to {outdir}")
