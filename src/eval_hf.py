"""Evaluate a (possibly unlearned) HF model on the forget/retain/heldout splits.

Generates solutions greedily for all 138 split problems, runs the official
HumanEval test cases, and writes eval_<tag>.jsonl plus a per-split summary
appended to summary.json.
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_humaneval import extract_code, run_tests

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
SYSTEM = (
    "You are an expert Python programmer. Complete the given function. "
    "Return the complete function implementation (including the signature and any "
    "needed imports) inside a single ```python code block. Do not include tests, "
    "examples, or explanations."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    args = ap.parse_args()

    out_path = f"eval_{args.tag}.jsonl"
    if os.path.exists(out_path) and sum(1 for _ in open(out_path)) == 138:
        print(f"{out_path} already complete, skipping generation")
    else:
        splits = json.load(open("splits.json"))
        problems = {p["task_id"]: p
                    for p in map(json.loads, open("HumanEval.jsonl"))}
        tasks = [(split, tid) for split in ("forget", "retain", "heldout")
                 for tid in splits[split]]

        tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa")
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
            model = model.merge_and_unload()
        model.eval()

        with open(out_path, "w") as out:
            for i in range(0, len(tasks), args.batch_size):
                chunk = tasks[i:i + args.batch_size]
                prompts = []
                for _, tid in chunk:
                    msgs = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content":
                             "Complete this function:\n\n```python\n"
                             + problems[tid]["prompt"] + "```"}]
                    prompts.append(tok.apply_chat_template(
                        msgs, add_generation_prompt=True, tokenize=False))
                enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    gen = model.generate(
                        **enc, do_sample=False, temperature=None, top_p=None,
                        top_k=None, max_new_tokens=args.max_new_tokens,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id)
                for (split, tid), seq in zip(chunk, gen):
                    text = tok.decode(seq[enc.input_ids.shape[1]:],
                                      skip_special_tokens=True)
                    p = problems[tid]
                    code = extract_code(text, p["prompt"], p["entry_point"])
                    passed, error = run_tests(code, p["test"], p["entry_point"])
                    out.write(json.dumps({
                        "task_id": tid, "split": split, "raw_response": text,
                        "generated_solution": code, "passed": passed,
                        "error": error}) + "\n")
                    out.flush()
                print(f"[{args.tag}] {min(i + args.batch_size, len(tasks))}"
                      f"/{len(tasks)}", flush=True)

    # Summarize.
    recs = [json.loads(l) for l in open(out_path)]
    summary = {}
    for split in ("forget", "retain", "heldout"):
        rs = [r for r in recs if r["split"] == split]
        summary[split] = {"passed": sum(r["passed"] for r in rs),
                          "total": len(rs),
                          "pass@1": round(sum(r["passed"] for r in rs)
                                          / len(rs), 4)}
    all_summaries = (json.load(open("summary.json"))
                     if os.path.exists("summary.json") else {})
    all_summaries[args.tag] = summary
    json.dump(all_summaries, open("summary.json", "w"), indent=2)
    print(json.dumps({args.tag: summary}, indent=2))


if __name__ == "__main__":
    main()
