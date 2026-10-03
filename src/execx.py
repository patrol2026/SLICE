"""Multilingual execution harness for HumanEval-X (Python / Java / C++).

Builds a runnable source file from (prompt + completion + test), compiles and
runs it in a temp dir with a timeout, and returns (passed, error).
"""

import os
import re
import subprocess
import tempfile

TIMEOUT = 20


def _py_entry(prompt):
    names = re.findall(r"def\s+(\w+)\s*\(", prompt)
    return names[-1] if names else "candidate"


COMMON_PY = ("from typing import *\nimport math, collections, itertools, "
             "heapq, bisect, re, functools, string\n")
COMMON_CPP = "#include<bits/stdc++.h>\nusing namespace std;\n"


def entry_name(lang, prompt):
    if lang == "python":
        m = re.findall(r"def\s+(\w+)\s*\(", prompt)
        return m[-1] if m else "candidate"
    if lang == "java":
        m = re.findall(r"public\s+[\w<>\[\], ]+?\s+(\w+)\s*\(", prompt)
        return m[-1] if m else ""
    if lang == "cpp":
        m = re.findall(r"(\w+)\s*\([^;]*\)\s*\{", prompt)
        return m[-1] if m else ""
    return ""


def extract_code(text, lang):
    blocks = re.findall(r"```[a-zA-Z+]*\s*\n(.*?)```", text, re.DOTALL)
    return blocks[0] if blocks else text


def build_source(lang, prompt, model_code, test):
    """Assemble a runnable file from the model's extracted code, robustly
    (handles both full-function/class outputs and body-only completions)."""
    ent = entry_name(lang, prompt)
    code = model_code
    if lang == "python":
        full = bool(re.search(r"def\s+%s\b" % re.escape(ent), code)) if ent else "def " in code
        body = (COMMON_PY + code) if full else (prompt + code)
        return body + "\n" + test + f"\ncheck({ent})\nprint('__PASS__')\n"
    if lang == "java":
        if "class Solution" in code:
            if "import java.util" not in code:
                code = "import java.util.*;\nimport java.lang.*;\n" + code
            return code + "\n" + test
        return prompt + code + "\n" + test
    if lang == "cpp":
        if "#include" in code or (ent and re.search(r"\b%s\s*\(" % re.escape(ent), code)):
            return COMMON_CPP + code + "\n" + test          # inject headers
        return COMMON_CPP + prompt + code + "\n" + test
    raise ValueError(lang)


def run_source(lang, src, timeout=TIMEOUT):
    return {"python": _run_py, "java": _run_java, "cpp": _run_cpp}[lang](src, timeout)


def run_x(lang, prompt, completion, test, timeout=TIMEOUT):
    """lang in {python, java, cpp}. completion = model/canonical body."""
    if lang == "python":
        src = prompt + completion + "\n" + test + \
            f"\ncheck({_py_entry(prompt)})\nprint('__PASS__')\n"
        return _run_py(src, timeout)
    if lang == "java":
        return _run_java(prompt + completion + "\n" + test, timeout)
    if lang == "cpp":
        return _run_cpp(prompt + completion + "\n" + test, timeout)
    raise ValueError(lang)


def _run_py(src, timeout):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        p = subprocess.run(["python3", path], capture_output=True, text=True,
                           timeout=timeout)
        return ("__PASS__" in p.stdout), (p.stderr or p.stdout)[-800:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)


def _run_java(src, timeout):
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "Main.java")
        open(path, "w").write(src)
        c = subprocess.run(["javac", path], capture_output=True, text=True,
                           timeout=60, cwd=d)
        if c.returncode != 0:
            return False, "compile:" + c.stderr[-800:]
        r = subprocess.run(["java", "-cp", d, "Main"], capture_output=True,
                           text=True, timeout=timeout, cwd=d)
        return (r.returncode == 0), (r.stderr or r.stdout)[-800:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        subprocess.run(["rm", "-rf", d])


def _run_cpp(src, timeout):
    d = tempfile.mkdtemp()
    try:
        cpp = os.path.join(d, "prog.cpp"); exe = os.path.join(d, "prog")
        open(cpp, "w").write(src)
        c = subprocess.run(["g++", "-std=c++17", "-O0", cpp, "-o", exe],
                           capture_output=True, text=True, timeout=60)
        if c.returncode != 0:
            return False, "compile:" + c.stderr[-800:]
        r = subprocess.run([exe], capture_output=True, text=True,
                           timeout=timeout)
        return (r.returncode == 0), (r.stderr or r.stdout)[-800:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        subprocess.run(["rm", "-rf", d])


if __name__ == "__main__":
    from datasets import load_dataset
    from concurrent.futures import ThreadPoolExecutor
    for lang in ["python", "java", "cpp"]:
        ds = load_dataset("THUDM/humaneval-x", lang, split="test",
                          trust_remote_code=True)
        pool = ThreadPoolExecutor(max_workers=16)
        def chk(r):
            return run_x(lang, r["prompt"], r["canonical_solution"], r["test"])[0]
        res = list(pool.map(chk, list(ds)))
        print(f"{lang}: canonical pass {sum(res)}/{len(res)}")
