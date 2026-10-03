"""Generate verified wrong-core mutants for the LeetCode SLICE pairs.

Forget pairs: mutate the important blocks of the 281 forget problems' GENERATED
solutions. Guard pairs: mutate the canonical solutions of 200 sampled retain
problems. A mutant is accepted only if the mutated full solution parses AND
fails the problem's tests. Output: lc_mutants.json
"""

import ast
import json
import os
import random

from gen_mutants import try_mutations
from run_leetcode import run_tests

N_GUARD = 200


def mutate_blocks(solution, blocks, test, entry_point):
    lines = solution.split("\n")
    out = []
    for b in blocks:
        s, e = b["start_line"] - 1, b["end_line"]
        original = "\n".join(lines[s:e])
        for cand in try_mutations(original):
            mutated_sol = "\n".join(lines[:s] + cand.split("\n") + lines[e:])
            try:
                ast.parse(mutated_sol)
            except SyntaxError:
                continue
            passed, _ = run_tests(mutated_sol, test, entry_point)
            if not passed:
                out.append({"start_line": b["start_line"],
                            "end_line": b["end_line"],
                            "original": original, "mutated": cand})
                break
    return out


def main():
    splits = json.load(open(os.environ.get("LC_SPLITS", "leetcode_splits.json")))
    forget_ids = set(splits["forget"])
    forget = [r for r in map(json.loads,
              open(os.environ.get("LC_FORGET_ANN", "leetcode_forget_generated_important_lines.jsonl")))
              if r["task_id"] in forget_ids and r.get("important_lines")]
    canon = {r["task_id"]: r
             for r in map(json.loads, open(os.environ.get("LC_CANON","leetcode_important_lines.jsonl")))}

    mutants = {"forget": {}, "guard": {}}
    n_f = 0
    for i, r in enumerate(forget):
        ms = mutate_blocks(r["generated_solution"], r["important_lines"],
                           r["test"], r["entry_point"])
        if ms:
            mutants["forget"][r["task_id"]] = ms
            n_f += len(ms)
        if (i + 1) % 50 == 0:
            print(f"forget {i + 1}/{len(forget)}: {n_f} mutant blocks", flush=True)

    rng = random.Random(0)
    guard_pool = [t for t in splits["retain"]
                  if t in canon and canon[t].get("important_lines")]
    guard_ids = rng.sample(guard_pool, min(N_GUARD, len(guard_pool)))
    n_g = 0
    for i, tid in enumerate(guard_ids):
        r = canon[tid]
        ms = mutate_blocks(r["completion"], r["important_lines"],
                           r["test"], r["entry_point"])
        if ms:
            mutants["guard"][tid] = ms
            n_g += len(ms)
        if (i + 1) % 50 == 0:
            print(f"guard {i + 1}/{N_GUARD}: {n_g} mutant blocks", flush=True)

    json.dump(mutants, open(os.environ.get("LC_MUTANTS_OUT", "lc_mutants.json"), "w"))
    print(f"DONE. forget: {n_f} blocks / {len(mutants['forget'])} tasks | "
          f"guard: {n_g} blocks / {len(mutants['guard'])} tasks")


if __name__ == "__main__":
    main()
