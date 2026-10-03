"""Evaluate a model on the LeetCode splits (1404 problems) with metrics.

Writes eval_lc_<tag>.jsonl (per-problem generation + pass/fail), a per-split
summary to lc_summary.json, and timing/token metrics to lc_metrics.json.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_leetcode import MODEL, RESULTS, SYSTEM, build_user_msg, extract_code, run_tests
ATTN = os.environ.get("LC_ATTN", "sdpa")
RDIR = os.environ.get("LC_RESULTS_DIR", ".")   # separate output dir per model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--splits", default="leetcode_splits.json")
    args = ap.parse_args()

    splits = json.load(open(args.splits))
    tasks = [(s, t) for s in ("forget", "heldout", "retain") for t in splits[s]]
    out_path = f"eval_lc_{args.tag}.jsonl"
    t0 = time.time()
    prompt_toks = gen_toks = 0

    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                done.add(json.loads(line)["task_id"])
            except json.JSONDecodeError:
                pass
    todo = [(s, t) for s, t in tasks if t not in done]
    if not todo:
        print(f"{out_path} already complete, skipping generation")
    else:
        print(f"{len(done)} done, {len(todo)} to go", flush=True)
        results = {r["task_id"]: r
                   for r in map(json.loads, open(RESULTS))}
        # sort by prompt length so batches have uniform padding
        todo.sort(key=lambda st: len(results[st[1]]["problem_description"]),
                  reverse=True)
        tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation=ATTN)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
            model = model.merge_and_unload()
        model.eval()
        pad = tok.pad_token_id or tok.eos_token_id

        pool = ThreadPoolExecutor(max_workers=args.batch_size)
        with open(out_path, "a") as out:
            for i in range(0, len(todo), args.batch_size):
                chunk = todo[i:i + args.batch_size]
                prompts = [tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": build_user_msg(results[t])}],
                    add_generation_prompt=True, tokenize=False)
                    for _, t in chunk]
                enc = tok(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=3072).to("cuda")
                prompt_toks += int(enc.attention_mask.sum().item())
                with torch.no_grad():
                    gen = model.generate(
                        **enc, do_sample=False, temperature=None, top_p=None,
                        top_k=None, max_new_tokens=args.max_new_tokens,
                        pad_token_id=pad)
                new = gen[:, enc.input_ids.shape[1]:]
                gen_toks += int((new != pad).sum().item())
                rows = []
                for (split, tid), seq in zip(chunk, new):
                    text = tok.decode(seq, skip_special_tokens=True)
                    code = extract_code(text)
                    rows.append([split, tid, text, code, results[tid]])
                futs = {}
                for j, (_, _, _, code, p) in enumerate(rows):
                    if "class Solution" in code:
                        futs[j] = pool.submit(run_tests, code, p["test"],
                                              p["entry_point"])
                for j, (split, tid, text, code, p) in enumerate(rows):
                    if j in futs:
                        passed, error = futs[j].result()
                    else:
                        passed, error = False, "no class Solution in output"
                    out.write(json.dumps({
                        "task_id": tid, "split": split,
                        "difficulty": p["difficulty"], "raw_response": text,
                        "generated_solution": code, "passed": passed,
                        "error": error}) + "\n")
                    out.flush()
                if (i // args.batch_size) % 10 == 0:
                    print(f"[{args.tag}] {len(done) + min(i + args.batch_size, len(todo))}"
                          f"/{len(tasks)}", flush=True)

    recs = [json.loads(l) for l in open(out_path)]
    summary = {}
    for split in ("forget", "heldout", "retain"):
        rs = [r for r in recs if r["split"] == split]
        summary[split] = {"passed": sum(r["passed"] for r in rs),
                          "total": len(rs),
                          "pass@1": round(sum(r["passed"] for r in rs) / len(rs), 4)}
    all_s = json.load(open(os.path.join(RDIR, "lc_summary.json"))) if os.path.exists(os.path.join(RDIR, "lc_summary.json")) else {}
    all_s[args.tag] = summary
    json.dump(all_s, open(os.path.join(RDIR, "lc_summary.json"), "w"), indent=1)

    all_m = json.load(open(os.path.join(RDIR, "lc_metrics.json"))) if os.path.exists(os.path.join(RDIR, "lc_metrics.json")) else {}
    all_m[f"eval_{args.tag}"] = {
        "eval_seconds": round(time.time() - t0, 1),
        "problems": len(tasks),
        "prompt_tokens": prompt_toks, "generated_tokens": gen_toks}
    json.dump(all_m, open(os.path.join(RDIR, "lc_metrics.json"), "w"), indent=1)
    print(json.dumps({args.tag: summary}, indent=1))


if __name__ == "__main__":
    main()
