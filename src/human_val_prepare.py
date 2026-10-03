"""Prepare blind human-validation sheets for important-line annotation.

Samples N problems (seed 42) from the LLM-annotated forget set and writes
identical blind sheets for annotators A and B under human_val/A and human_val/B.
LLM labels are NOT included in the sheets.

Usage: python human_val_prepare.py [N=100]
"""

import json
import random
import sys
from pathlib import Path

SRC = "leetcode_forget_generated_important_lines.jsonl"
SEED = 42

GUIDELINES = """\
IMPORTANT-LINE ANNOTATION GUIDELINES (v1 - frozen)

Definition
  A line is IMPORTANT if it encodes problem-specific core logic: a predicate,
  computation, or state update whose alteration would change the algorithm's
  output on the problem's tests.
  A line is NOT important if it is scaffolding: imports, function/class
  signatures, docstrings/comments, variable initialisation to trivial values,
  plain returns of an already-computed result, or boilerplate control flow
  that any solution would share.

Rules
  1. Judge each line on its own; typically 20-40% of content lines qualify,
     but there is NO quota - mark exactly what fits the definition.
  2. Blank lines and comment-only lines are never marked.
  3. A multi-line logical unit (e.g. a condition split over two lines) may be
     marked wholly.
  4. Do not consult the other annotator, any model, or the LLM labels.
  5. Do not revise answers after Step 3 (agreement) has been computed.

How to answer
  In each sheet, fill the final ANSWER line with the important line numbers,
  comma-separated (ranges allowed), e.g.:  ANSWER: 5,7-9,12
  If you judge that NO line qualifies, write:  ANSWER: none
"""


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    recs = [json.loads(l) for l in open(SRC)]
    recs = [r for r in recs if r.get("passed")]
    random.Random(SEED).shuffle(recs)
    sample = sorted(recs[:n], key=lambda r: r["task_id"])

    base = Path("human_val")
    for who in ("A", "B"):
        (base / who).mkdir(parents=True, exist_ok=True)
    (base / "GUIDELINES.txt").write_text(GUIDELINES)

    manifest = []
    for k, r in enumerate(sample, 1):
        desc = r["problem_description"].strip().split("\n")
        desc = "\n".join(desc[:25])
        lines = r["generated_solution"].split("\n")
        numbered = "\n".join(f"{i:3d}| {ln}" for i, ln in enumerate(lines, 1))
        sheet = (f"# problem {k}/{len(sample)}  task_id: {r['task_id']}\n"
                 f"# ---------------- PROBLEM ----------------\n{desc}\n"
                 f"# ---------------- SOLUTION ----------------\n{numbered}\n"
                 f"#\n# Mark the IMPORTANT line numbers (see GUIDELINES.txt).\n"
                 f"# Example format ->  ANSWER: 5,7-9,12\n"
                 f"ANSWER: \n")
        fname = f"{k:03d}_{r['task_id']}.txt"
        for who in ("A", "B"):
            (base / who / fname).write_text(sheet)
        manifest.append({"idx": k, "task_id": r["task_id"],
                         "n_lines": len(lines)})
    json.dump({"seed": SEED, "n": len(sample), "source": SRC,
               "problems": manifest},
              open(base / "manifest.json", "w"), indent=1)
    print(f"wrote {len(sample)} sheets to human_val/A and human_val/B "
          f"(seed {SEED}); guidelines in human_val/GUIDELINES.txt")


if __name__ == "__main__":
    main()
