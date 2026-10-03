"""Completion-style HumanEval (no chat template) for the prod-task models.

Matches PROD's own utility protocol: raw function prompt, greedy
continuation, truncate at the next top-level statement, execute tests.
Usage: python prod_completion_util.py --tag T [--adapters a,b]
Appends {tag: pass@1} under 'humaneval_completion' in utility_results.json.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_humaneval import run_tests

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
B = 16
STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@"]


def truncate(cont):
    cut = len(cont)
    for s in STOPS:
        i = cont.find(s)
        if i != -1:
            cut = min(cut, i)
    return cont[:cut]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapters", default=None)
    args = ap.parse_args()
    probs = [json.loads(l) for l in open("HumanEval.jsonl")]

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    from peft import PeftModel
    for a in (args.adapters.split(",") if args.adapters else []):
        model = PeftModel.from_pretrained(model, a)
        model = model.merge_and_unload()
    model.eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    pool = ThreadPoolExecutor(max_workers=32)
    futs = []
    for i in range(0, len(probs), B):
        chunk = probs[i:i + B]
        enc = tok([p["prompt"] for p in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=1024).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=384,
                                 pad_token_id=pad)
        for p, seq in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
            cont = truncate(tok.decode(seq, skip_special_tokens=True))
            code = p["prompt"] + cont
            futs.append(pool.submit(run_tests, code, p["test"], p["entry_point"]))
    n = sum(f.result()[0] for f in futs)
    score = round(n / len(probs), 4)
    allr = json.load(open("utility_results.json"))
    allr.setdefault(args.tag, {})["humaneval_completion"] = score
    json.dump(allr, open("utility_results.json", "w"), indent=1)
    print(f"[{args.tag}] humaneval_completion {score}")


if __name__ == "__main__":
    main()
