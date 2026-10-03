"""Build the human-validation annotation package (CSV format).

- Samples 100 problems (seed 42) from the Qwen ∩ DeepSeek solved pool,
  stratified by difficulty, split 50 selection / 50 confirmation.
- Writes:
    human_val/ANNOTATION_GUIDELINES.md
    human_val/annotation_A.csv   (annotator A - fill 'important' column)
    human_val/annotation_B.csv   (identical blind copy for annotator B)
    human_val/problems_reference.txt  (problem statements, for context)
    human_val/manifest_csv.json
CSV rows are individual solution lines; blank/comment lines are pre-marked
'skip' and must not be labeled.
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
N = 100
BASE = Path("human_val")

GUIDELINES = """\
# Important-Line Annotation Guidelines (v1 — frozen)

## Task
For each solution line in your CSV, decide whether the line is **IMPORTANT**
(encodes problem-specific core logic) and put `1` in the `important` column,
or leave it `0`.

## Definition
A line is **IMPORTANT** if it encodes problem-specific core logic — a
predicate, computation, or state update whose alteration would change the
algorithm's output on the problem's tests.

A line is **NOT important** if it is scaffolding — code that any solution to
any problem might share:
- imports, class/function signatures
- docstrings and comments
- trivial initialisation (`d = {}`, `count = 0`, `result = []`)
- plain returns of an already-computed value (`return result`)
- boilerplate control flow every solution would have (e.g., a bare loop header
  that merely iterates the input **unless** its bounds/order encode the trick)

## Decision rules
1. Judge each line by the question: *"if an adversary corrupted only this
   line, would the solution break on this problem's tests in a
   problem-specific way?"*
2. Typically 20–40% of content lines qualify — but there is **no quota**;
   mark exactly what fits the definition.
3. Rows pre-marked `skip` (blank lines, comment-only lines) must be left
   untouched.
4. A multi-line logical unit (a condition split across lines) may be marked
   wholly.
5. Work **independently**: do not consult the other annotator, any AI model,
   or any external solution. Do not revise answers after submitting.
6. Read the problem statement (problems_reference.txt) before judging its
   lines — importance is relative to *this* problem.

## Worked example
Problem: return True if any two numbers in the list are closer than
`threshold`.

```
 1| from typing import List                                  -> 0 (import)
 2| def has_close_elements(numbers, threshold):              -> 0 (signature)
 3|     for i in range(len(numbers)):                        -> 0 (bare iteration)
 4|         for j in range(i + 1, len(numbers)):             -> 1 (i+1 pairing logic)
 5|             if abs(numbers[i] - numbers[j]) < threshold: -> 1 (core predicate)
 6|                 return True                              -> 0 (plain return)
 7|     return False                                         -> 0 (plain return)
```
Line 4 is marked because `i + 1` encodes the pair-enumeration trick; line 5 is
the problem's defining comparison. Lines 3, 6, 7 appear in countless
solutions to countless problems.

## Procedure
1. Open your own CSV (annotation_A.csv or annotation_B.csv). Never open the
   other annotator's file.
2. For each problem (grouped rows), read its statement in
   problems_reference.txt, then fill `important` with 1 or 0 for every row
   not marked `skip`.
3. Expect ~2 minutes per problem, ~3.5 hours total. Take breaks; do not
   annotate when tired.
4. When both files are complete, run: `python human_val_score_csv.py agree`
"""


def content(ln):
    s = ln.strip()
    return bool(s) and not s.startswith("#")


def main():
    solved = {}
    for l in open("results_leetcode.jsonl"):
        r = json.loads(l)
        if r.get("passed"):
            solved[r["task_id"]] = r
    ds_ok = {json.loads(l)["task_id"] for l in open("results_leetcode_dsv2.jsonl")
             if json.loads(l).get("passed")}
    pool = [r for t, r in solved.items() if t in ds_ok]

    # stratified sample by difficulty, proportional
    by_diff = defaultdict(list)
    for r in pool:
        by_diff[r["difficulty"]].append(r)
    rng = random.Random(SEED)
    sample = []
    total = len(pool)
    for d in sorted(by_diff):
        k = round(N * len(by_diff[d]) / total)
        rng.shuffle(by_diff[d])
        sample += by_diff[d][:k]
    sample = sample[:N]
    rng.shuffle(sample)                      # order-mix difficulties
    halves = {r["task_id"]: ("selection" if i < N // 2 else "confirmation")
              for i, r in enumerate(sample)}
    sample = sorted(sample, key=lambda r: r["task_id"])

    BASE.mkdir(exist_ok=True)
    (BASE / "ANNOTATION_GUIDELINES.md").write_text(GUIDELINES)

    header = ["problem_no", "task_id", "difficulty", "line_no", "code", "important"]
    rows, ref, n_content = [], [], 0
    for k, r in enumerate(sample, 1):
        ref.append(f"{'='*70}\nproblem {k:03d}  task_id: {r['task_id']}  "
                   f"({r['difficulty']})\n{'='*70}\n"
                   f"{r['problem_description'].strip()}\n")
        for i, ln in enumerate(r["generated_solution"].split("\n"), 1):
            lab = "" if content(ln) else "skip"
            n_content += lab == ""
            rows.append([k, r["task_id"], r["difficulty"], i, ln, lab])

    for who in ("A", "B"):
        with open(BASE / f"annotation_{who}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    (BASE / "problems_reference.txt").write_text("\n".join(ref))

    stats = defaultdict(int)
    for r in sample:
        stats[r["difficulty"]] += 1
    json.dump({"seed": SEED, "n": len(sample),
               "pool": "qwen solved ∩ deepseek solved",
               "pool_size": len(pool), "difficulty_mix": dict(stats),
               "content_lines": n_content,
               "halves": halves},
              open(BASE / "manifest_csv.json", "w"), indent=1)
    print(f"pool (both models solve): {len(pool)}")
    print(f"sampled {len(sample)} problems, difficulty mix {dict(stats)}")
    print(f"content-line decisions: {n_content}")
    print("wrote human_val/annotation_A.csv, annotation_B.csv, "
          "problems_reference.txt, ANNOTATION_GUIDELINES.md")


if __name__ == "__main__":
    main()
