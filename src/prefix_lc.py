"""Prefix-injection attack for the LeetCode task (analogue of the copyrighted
prefix attack). For each forget problem we prime the assistant turn with the
first 25/50/75% of the model's own (forgotten) solution and let it complete,
then run the tests. A high completed-pass@1 means the forgotten capability can
be re-elicited by priming; a low one means the removal is prefix-robust.

Writes prefix_<tag> -> LC_RESULTS_DIR/prefix_results.json.
Usage: python prefix_lc.py --tag T [--adapter A]
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_leetcode import MODEL, RESULTS, SYSTEM, build_user_msg, extract_code, run_tests

SPLITS = os.environ.get("LC_SPLITS", "leetcode_splits_s10.json")
RDIR = os.environ.get("LC_RESULTS_DIR", ".")
FRACS = [0.25, 0.50, 0.75]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    splits = json.load(open(SPLITS))
    results = {r["task_id"]: r for r in map(json.loads, open(RESULTS))}
    forget = [t for t in splits["forget"]
              if t in results and results[t].get("passed")]

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()
    pad = tok.pad_token_id

    B = int(os.environ.get("LC_GEN_BATCH", "10"))
    res = {}
    for frac in FRACS:
        prompts, metas = [], []
        for t in forget:
            sol = results[t]["generated_solution"]
            lines = sol.split("\n")
            h = max(1, int(len(lines) * frac))
            prime = "\n".join(lines[:h])
            chat = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": build_user_msg(results[t])}],
                add_generation_prompt=True, tokenize=False)
            prompts.append(chat + "```python\n" + prime + "\n")
            metas.append((t, prime))
        passed = total = 0
        for i in range(0, len(prompts), B):
            chunk, mchunk = prompts[i:i + B], metas[i:i + B]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=3072).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=False, temperature=None,
                                     top_p=None, top_k=None, max_new_tokens=768,
                                     pad_token_id=pad)
            for (t, prime), seq in zip(mchunk, gen):
                cont = tok.decode(seq[enc.input_ids.shape[1]:],
                                  skip_special_tokens=True)
                full = "```python\n" + prime + "\n" + cont
                code = extract_code(full) or (prime + "\n" + cont)
                ok, _ = run_tests(code, results[t]["test"],
                                  results[t]["entry_point"])
                passed += int(ok)
                total += 1
        res[f"prefix_{int(frac * 100)}"] = round(passed / max(total, 1), 4)
        print(f"[{args.tag}] frac {frac}: pass@1 {res[f'prefix_{int(frac*100)}']}",
              flush=True)

    path = os.path.join(RDIR, "prefix_results.json")
    allr = json.load(open(path)) if os.path.exists(path) else {}
    allr[args.tag] = res
    json.dump(allr, open(path, "w"), indent=1)
    print(json.dumps({args.tag: res}))


if __name__ == "__main__":
    main()
