"""Generate verified wrong-core mutants of the important-line blocks.

For every forget and retain problem, mutate each annotated important block with
mutation-testing style operator flips. A mutant is accepted only if the full
mutated solution still parses AND fails the official HumanEval tests.
Output: mutants.json  {task_id: [{block_idx, original, mutated}, ...]}
"""

import ast
import json
import re

from run_humaneval import run_tests

MUTATIONS = [
    (r'(?<![<>=!])<=(?!=)', '>'),
    (r'(?<![<>=!])>=(?!=)', '<'),
    (r'(?<![<>=!])<(?![<=])', '>='),
    (r'(?<![<>=!])>(?![>=])', '<='),
    (r'==', '!='),
    (r'!=', '=='),
    (r'\band\b', 'or'),
    (r'\bor\b', 'and'),
    (r'\bmax\b', 'min'),
    (r'\bmin\b', 'max'),
    (r'\babs\(', '('),
    (r'\+ 1\b', '- 1'),
    (r'- 1\b', '+ 1'),
    (r'(?<![+*/-])\+(?![+=])', '-'),
    (r'(?<![+*/-])-(?![-=>])', '+'),
    (r'(?<![*/])\*(?![*=])', '+'),
    (r'%', '//'),
    (r'\bTrue\b', 'False'),
    (r'\bFalse\b', 'True'),
    (r'\b0\b', '1'),
    (r'\b1\b', '2'),
    (r'\b2\b', '3'),
]


def try_mutations(block):
    for pat, rep in MUTATIONS:
        m = re.search(pat, block)
        if m:
            yield re.sub(pat, rep, block, count=1)


def main():
    splits = json.load(open("splits.json"))
    results = {r["task_id"]: r for r in map(
        json.loads, open("results_qwen2.5-coder_7b_important_lines.jsonl"))}
    problems = {p["task_id"]: p for p in map(json.loads, open("HumanEval.jsonl"))}

    mutants, failures = {}, []
    for tid in splits["forget"] + splits["retain"]:
        r, p = results[tid], problems[tid]
        lines = r["generated_solution"].split("\n")
        out = []
        for bi, b in enumerate(r["important_lines"]):
            s, e = b["start_line"] - 1, b["end_line"]
            original = "\n".join(lines[s:e])
            accepted = None
            for cand in try_mutations(original):
                mutated_sol = "\n".join(lines[:s] + cand.split("\n") + lines[e:])
                try:
                    ast.parse(mutated_sol)
                except SyntaxError:
                    continue
                passed, _ = run_tests(mutated_sol, p["test"], p["entry_point"])
                if not passed:
                    accepted = cand
                    break
            if accepted is None:
                failures.append((tid, bi, original))
            else:
                out.append({"block_idx": bi, "start_line": b["start_line"],
                            "end_line": b["end_line"], "original": original,
                            "mutated": accepted})
        mutants[tid] = out

    json.dump(mutants, open("mutants.json", "w"), indent=1)
    n_blocks = sum(len(v) for v in mutants.values())
    print(f"mutated {n_blocks} blocks across {len(mutants)} tasks; "
          f"{len(failures)} blocks unmutated")
    for tid, bi, orig in failures:
        print(f"  FAILED {tid} block {bi}: {orig[:80]!r}")


if __name__ == "__main__":
    main()
