"""Few-shot LLM important-line detector.

Prompts the local code model with 3 worked examples drawn from HumanEval
(NOT LeetCode, so no overlap with the evaluation set) that demonstrate marking
the problem-specific important lines, then asks it to mark each forget solution.

Outputs fewshot_detections.jsonl and merges metrics into iline_stats.json.
"""

import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_detectors import (FORGET, CODE_MODEL, ref_lines, prf, kappa)

# --- 3 hand-built demonstrations from HumanEval (out-of-distribution vs LeetCode) ---
FEWSHOT = """You mark the IMPORTANT LINES of a Python solution: the lines with the
problem-specific core logic (key predicate, formula, recurrence, decision), NOT
generic scaffolding (imports, def, docstrings, trivial init, bare return).
Reply ONLY with the important line numbers as a comma-separated list.

### Example 1
  1| from typing import List
  2| def has_close_elements(numbers: List[float], threshold: float) -> bool:
  3|     for i in range(len(numbers)):
  4|         for j in range(i + 1, len(numbers)):
  5|             if abs(numbers[i] - numbers[j]) < threshold:
  6|                 return True
  7|     return False
IMPORTANT: 5, 6

### Example 2
  1| def greatest_common_divisor(a: int, b: int) -> int:
  2|     while b:
  3|         a, b = b, a % b
  4|     return a
IMPORTANT: 2, 3

### Example 3
  1| def sum_squares(lst):
  2|     result = 0
  3|     for i in range(len(lst)):
  4|         if i % 3 == 0:
  5|             result += lst[i] ** 2
  6|         elif i % 4 == 0:
  7|             result += lst[i] ** 3
  8|         else:
  9|             result += lst[i]
 10|     return result
IMPORTANT: 4, 5, 6, 7, 9
"""


def parse_lines(text, n_lines):
    m = re.search(r"IMPORTANT:\s*([0-9,\s]+)", text)
    if not m:
        m = re.search(r"([0-9]+(?:\s*,\s*[0-9]+)+)", text)
    if not m:
        return set()
    out = set()
    for tok in m.group(1).split(","):
        tok = tok.strip()
        if tok.isdigit() and 1 <= int(tok) <= n_lines:
            out.add(int(tok))
    return out


def main():
    recs = [json.loads(l) for l in open(FORGET)]
    tok = AutoTokenizer.from_pretrained(CODE_MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(CODE_MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    pad = tok.pad_token_id or tok.eos_token_id

    dets, f1s, ious, kaps = [], [], [], []
    t0 = time.time()
    B = 16
    for i in range(0, len(recs), B):
        chunk = recs[i:i + B]
        prompts = []
        for rec in chunk:
            lines = rec["generated_solution"].split("\n")
            numbered = "\n".join(f"{j+1:3d}| {ln}" for j, ln in enumerate(lines))
            user = (FEWSHOT + f"\n### Now mark this solution\n{numbered}\nIMPORTANT:")
            prompts.append(tok.apply_chat_template(
                [{"role": "user", "content": user}],
                add_generation_prompt=True, tokenize=False))
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=40,
                                 pad_token_id=pad)
        for rec, seq in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
            out = tok.decode(seq, skip_special_tokens=True)
            n = len(rec["generated_solution"].split("\n"))
            pred = parse_lines("IMPORTANT: " + out, n)
            ref = ref_lines(rec)
            p, r, f, iou = prf(pred, ref)
            f1s.append(f); ious.append(iou)
            kaps.append(kappa(pred, ref, n))
            dets.append({"task_id": rec["task_id"], "detected": sorted(pred),
                         "ref_lines": sorted(ref)})
        print(f"[fewshot] {min(i+B,len(recs))}/{len(recs)}", flush=True)

    with open("fewshot_detections.jsonl", "w") as fo:
        for d in dets:
            fo.write(json.dumps(d) + "\n")

    def avg(x): return round(sum(x) / len(x), 4)
    stat = {"F1_vs_LLM": avg(f1s), "IoU_vs_LLM": avg(ious),
            "kappa_vs_LLM": avg(kaps),
            "time_sec_total": round(time.time() - t0, 1),
            "time_ms_per_problem": round(1000 * (time.time() - t0) / len(recs), 1),
            "needs_model": True, "needs_tests": False, "few_shot": True}
    alls = json.load(open("iline_stats.json")) if __import__("os").path.exists("iline_stats.json") else {}
    alls["fewshot_llm"] = stat
    json.dump(alls, open("iline_stats.json", "w"), indent=1)
    print(json.dumps({"fewshot_llm": stat}, indent=1))


if __name__ == "__main__":
    main()
