"""Paraphrase probe: evaluate a model on paraphrased forget-problem statements.

Same starter code, entry point, and tests — only the problem description is
reworded. If the unlearned model solves paraphrases but not originals, the
forgetting is prompt-anchored. Results appended to paraphrase_results.json.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_leetcode import MODEL, RESULTS, SYSTEM, build_user_msg, extract_code, run_tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("LC_GEN_BATCH","32")))
    args = ap.parse_args()

    paras = json.load(open(os.environ.get("LC_PARAS","lc_paraphrases.json")))
    results = {r["task_id"]: r
               for r in map(json.loads, open(RESULTS))}
    tasks = sorted(paras)

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()
    pad = tok.pad_token_id or tok.eos_token_id
    pool = ThreadPoolExecutor(max_workers=args.batch_size)

    recs = []
    for i in range(0, len(tasks), args.batch_size):
        chunk = tasks[i:i + args.batch_size]
        prompts = []
        for t in chunk:
            rec = dict(results[t])
            rec["problem_description"] = paras[t]
            prompts.append(tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": build_user_msg(rec)}],
                add_generation_prompt=True, tokenize=False))
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=3072).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, max_new_tokens=768,
                                 pad_token_id=pad)
        rows = []
        for t, seq in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
            code = extract_code(tok.decode(seq, skip_special_tokens=True))
            rows.append((t, code))
        futs = {t: pool.submit(run_tests, c, results[t]["test"],
                               results[t]["entry_point"])
                for t, c in rows if "class Solution" in c}
        for t, c in rows:
            passed, err = futs[t].result() if t in futs else (False, "no class")
            recs.append({"task_id": t, "passed": passed})
        print(f"[{args.tag}] {len(recs)}/{len(tasks)}", flush=True)

    rate = sum(r["passed"] for r in recs) / len(recs)
    all_r = (json.load(open("paraphrase_results.json"))
             if os.path.exists("paraphrase_results.json") else {})
    all_r[args.tag] = {"paraphrase_pass@1": round(rate, 4),
                       "n": len(recs),
                       "per_task": {r["task_id"]: r["passed"] for r in recs}}
    json.dump(all_r, open("paraphrase_results.json", "w"))
    print(f"{args.tag}: paraphrase pass@1 = {rate:.1%}")


if __name__ == "__main__":
    main()
