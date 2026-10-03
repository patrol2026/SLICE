"""Create forget/retain/held-out splits from the correctly solved HumanEval problems."""

import json
import random

RESULTS = "results_qwen2.5-coder_7b.jsonl"

records = [json.loads(l) for l in open(RESULTS)]
correct = sorted([r["task_id"] for r in records if r["passed"]],
                 key=lambda t: int(t.split("/")[1]))
assert len(correct) == 138, len(correct)

rng = random.Random(42)
rng.shuffle(correct)

splits = {
    "forget": sorted(correct[:70], key=lambda t: int(t.split("/")[1])),
    "retain": sorted(correct[70:100], key=lambda t: int(t.split("/")[1])),
    "heldout": sorted(correct[100:], key=lambda t: int(t.split("/")[1])),
}
json.dump(splits, open("splits.json", "w"), indent=2)
print({k: len(v) for k, v in splits.items()})
