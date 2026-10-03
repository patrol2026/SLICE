"""Run Qwen2.5-Coder-7B-Instruct over LeetCodeDataset and test every solution.

Generates greedily (HF bf16, batched), extracts the class Solution code, and runs
the dataset's check(candidate) asserts in a sandboxed subprocess. Results are
appended incrementally to results_leetcode.jsonl (resumable).
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("LC_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
RESULTS = os.environ.get("LC_RESULTS", "results_leetcode.jsonl")
BATCH = 12
MAX_NEW = 1024
TIMEOUT_S = 25

SYSTEM = (
    "You are an expert Python programmer. Solve the given LeetCode problem. "
    "Return the complete solution (the full class Solution with any needed "
    "imports) inside a single ```python code block. Do not include tests, "
    "examples, or explanations."
)

# Standard prelude used when executing solutions (LeetCode-style imports).
PRELUDE = (
    "from typing import *\n"
    "from collections import *\n"
    "from itertools import *\n"
    "from functools import *\n"
    "from heapq import *\n"
    "from bisect import *\n"
    "import math, re, string, random\n"
    "from math import *\n\n"
)


def build_user_msg(p):
    return (p["problem_description"].strip()
            + "\n\nComplete this starter code:\n\n```python\n"
            + p["starter_code"].strip() + "\n```")


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return blocks[0] if blocks else text


def run_tests(code, test, entry_point):
    program = (PRELUDE + code + "\n\n" + test
               + f"\n\ncheck({entry_point})\nprint('__PASS__')\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True,
                              text=True, timeout=TIMEOUT_S)
        if "__PASS__" in proc.stdout:
            return True, ""
        return False, (proc.stderr or proc.stdout)[-1500:]
    except subprocess.TimeoutExpired:
        return False, f"timeout after {TIMEOUT_S}s"
    finally:
        os.unlink(path)


def main():
    ds = load_dataset("newfacade/LeetCodeDataset")
    problems = list(ds["train"]) + list(ds["test"])

    done = set()
    n_pass = n_total = 0
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            try:
                r = json.loads(line)
                done.add(r["task_id"])
                n_total += 1
                n_pass += r["passed"]
            except json.JSONDecodeError:
                pass
    todo = [p for p in problems if p["task_id"] not in done]
    print(f"{len(problems)} problems, {len(done)} done, {len(todo)} to go", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    with open(RESULTS, "a") as out:
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": build_user_msg(p)}],
                add_generation_prompt=True, tokenize=False) for p in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=3072).to("cuda")
            with torch.no_grad():
                gen = model.generate(
                    **enc, do_sample=False, temperature=None, top_p=None,
                    top_k=None, max_new_tokens=MAX_NEW,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for p, seq in zip(chunk, gen):
                text = tok.decode(seq[enc.input_ids.shape[1]:],
                                  skip_special_tokens=True)
                code = extract_code(text)
                if "class Solution" not in code:
                    passed, error = False, "no class Solution in output"
                else:
                    passed, error = run_tests(code, p["test"], p["entry_point"])
                n_total += 1
                n_pass += passed
                out.write(json.dumps({
                    "task_id": p["task_id"], "difficulty": p["difficulty"],
                    "estimated_date": str(p["estimated_date"]),
                    "entry_point": p["entry_point"],
                    "problem_description": p["problem_description"],
                    "starter_code": p["starter_code"],
                    "test": p["test"],
                    "raw_response": text, "generated_solution": code,
                    "passed": passed, "error": error}) + "\n")
                out.flush()
            print(f"[{n_total}/{len(problems)}] pass@1 so far: "
                  f"{n_pass}/{n_total} = {n_pass/n_total:.1%}", flush=True)

    print(f"\nDONE. pass@1 = {n_pass}/{n_total} = {n_pass/n_total:.1%}")


if __name__ == "__main__":
    main()
