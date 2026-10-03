"""Shared helpers for the CyberSecEval pipeline."""
import re


def vuln_index(code, line_text):
    """1-based index of the vulnerable line within the snippet."""
    t = line_text.strip()
    if not t:
        return None
    lines = code.split("\n")
    for i, ln in enumerate(lines, 1):
        if ln.strip() == t:
            return i
    for i, ln in enumerate(lines, 1):
        if t in ln:
            return i
    return None


def prompt_target(code, idx, min_prompt=2):
    """Split the snippet so the model must GENERATE the vulnerable line:
    prompt = lines before the vuln line, target = vuln line onward."""
    lines = code.split("\n")
    cut = max(min_prompt, idx - 1)
    prompt = "\n".join(lines[:cut]) + "\n"
    target = "\n".join(lines[cut:])
    return prompt, target


def vuln_fires(rule, text):
    try:
        return bool(re.search(rule, text))
    except re.error:
        return False
