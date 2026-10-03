"""Run HumanEval with qwen2.5-coder:7b-instruct via Ollama and evaluate solutions.

For each of the 164 HumanEval problems:
  1. Ask the model to complete the function.
  2. Extract the code from the response.
  3. Run the official HumanEval test cases (check(entry_point)) in a subprocess with a timeout.
  4. Append the result (generated solution + pass/fail) to results_qwen2.5-coder_7b.jsonl.

Resumable: already-completed task_ids in the results file are skipped.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time

import requests

MODEL = "qwen2.5-coder:7b-instruct"
OLLAMA_URL = "http://localhost:11434/api/chat"
DATASET = "HumanEval.jsonl"
RESULTS = "results_qwen2.5-coder_7b.jsonl"
TIMEOUT_S = 15

SYSTEM = (
    "You are an expert Python programmer. Complete the given function. "
    "Return the complete function implementation (including the signature and any "
    "needed imports) inside a single ```python code block. Do not include tests, "
    "examples, or explanations."
)


def generate(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Complete this function:\n\n```python\n" + prompt + "```"},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 1536},
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def extract_code(text: str, prompt: str, entry_point: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    code = blocks[0] if blocks else text
    # If the model returned only a function body, prepend the original prompt.
    if f"def {entry_point}" not in code:
        code = prompt + "\n" + code
    return code


def run_tests(code: str, test: str, entry_point: str) -> tuple[bool, str]:
    program = (
        code
        + "\n\n"
        + test
        + f"\n\ncheck({entry_point})\nprint('__PASS__')\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
        if "__PASS__" in proc.stdout:
            return True, ""
        return False, (proc.stderr or proc.stdout)[-2000:]
    except subprocess.TimeoutExpired:
        return False, f"timeout after {TIMEOUT_S}s"
    finally:
        os.unlink(path)


def main() -> None:
    problems = [json.loads(l) for l in open(DATASET)]

    done = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            try:
                done.add(json.loads(line)["task_id"])
            except json.JSONDecodeError:
                pass

    n_pass = n_total = 0
    if done:
        for line in open(RESULTS):
            try:
                r = json.loads(line)
                n_total += 1
                n_pass += r["passed"]
            except json.JSONDecodeError:
                pass

    with open(RESULTS, "a") as out:
        for p in problems:
            if p["task_id"] in done:
                continue
            t0 = time.time()
            raw = generate(p["prompt"])
            code = extract_code(raw, p["prompt"], p["entry_point"])
            passed, error = run_tests(code, p["test"], p["entry_point"])
            n_total += 1
            n_pass += passed
            record = {
                "task_id": p["task_id"],
                "entry_point": p["entry_point"],
                "prompt": p["prompt"],
                "raw_response": raw,
                "generated_solution": code,
                "passed": passed,
                "error": error,
                "gen_time_s": round(time.time() - t0, 1),
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(
                f"[{n_total}/164] {p['task_id']}: {'PASS' if passed else 'FAIL'} "
                f"(pass@1 so far: {n_pass}/{n_total} = {n_pass/n_total:.1%})",
                flush=True,
            )

    print(f"\nDONE. pass@1 = {n_pass}/{n_total} = {n_pass/n_total:.1%}")


if __name__ == "__main__":
    main()
