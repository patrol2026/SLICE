"""Create the two split protocols for both models from their solved pools.

Protocols (fractions of the solved pool, seed 42):
  A = 10/20/70  (forget/heldout/retain)
  B = 20/20/60
Existing cells are reused untouched:
  qwen B  = leetcode_splits.json      (281/281/842)
  ds   A  = leetcode_splits_ds.json   (107/213/747)
New cells written here:
  splits_qwen_A.json   (qwen 10/20/70)
  splits_ds_B.json     (ds   20/20/60)
"""

import json
import random

SEED = 42


def make(res_file, fr, hd, out):
    solved = sorted(json.loads(l)["task_id"] for l in open(res_file)
                    if json.loads(l).get("passed"))
    rng = random.Random(SEED)
    rng.shuffle(solved)
    n = len(solved)
    nf, nh = round(fr * n), round(hd * n)
    splits = {"forget": sorted(solved[:nf]),
              "heldout": sorted(solved[nf:nf + nh]),
              "retain": sorted(solved[nf + nh:])}
    json.dump(splits, open(out, "w"))
    print(out, {k: len(v) for k, v in splits.items()}, f"of {n}")


if __name__ == "__main__":
    make("results_leetcode.jsonl", 0.10, 0.20, "splits_qwen_A.json")
    make("results_leetcode_dsv2.jsonl", 0.20, 0.20, "splits_ds_B.json")
