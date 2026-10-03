"""Post-unlearning utility suite for a (possibly adapted) model:
  - HumanEval (164) and MBPP-sanitized test split: cross-benchmark code pass@1
  - MMLU (500-question sample): 4-way choice accuracy via answer-letter logprob
  - GSM8K (200-question sample): exact-match accuracy on the final number
Appends to utility_results.json under --tag.
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_humaneval import extract_code as he_extract, run_tests as he_run
from run_leetcode import MODEL
RDIR = os.environ.get("LC_RESULTS_DIR", ".")   # separate output dir per model

HE_SYSTEM = ("You are an expert Python programmer. Complete the given function. "
             "Return the complete function implementation inside a single "
             "```python code block. Do not include tests or explanations.")


def load_model(adapter):
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        for a in adapter.split(","):
            model = PeftModel.from_pretrained(model, a)
            model = model.merge_and_unload()
    model.eval()
    return tok, model


def batched_generate(tok, model, user_msgs, max_new=768, bs=int(os.environ.get("LC_GEN_BATCH","32")), system=HE_SYSTEM):
    pad = tok.pad_token_id or tok.eos_token_id
    outs = []
    for i in range(0, len(user_msgs), bs):
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": m}],
            add_generation_prompt=True, tokenize=False)
            for m in user_msgs[i:i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=3072).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, max_new_tokens=max_new,
                                 pad_token_id=pad)
        outs += [tok.decode(s, skip_special_tokens=True)
                 for s in gen[:, enc.input_ids.shape[1]:]]
    return outs


def eval_humaneval(tok, model, pool):
    probs = [json.loads(l) for l in open("HumanEval.jsonl")]
    msgs = ["Complete this function:\n\n```python\n" + p["prompt"] + "```"
            for p in probs]
    outs = batched_generate(tok, model, msgs)
    futs = []
    for p, o in zip(probs, outs):
        code = he_extract(o, p["prompt"], p["entry_point"])
        futs.append(pool.submit(he_run, code, p["test"], p["entry_point"]))
    n = sum(f.result()[0] for f in futs)
    return round(n / len(probs), 4)


def eval_mbpp(tok, model, pool):
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    msgs, metas = [], []
    for ex in ds:
        m = re.search(r"assert\s+(\w+)\s*\(", ex["test_list"][0])
        if not m:
            continue
        fname = m.group(1)
        msgs.append(f"{ex['prompt']}\nName the function `{fname}`. Return the "
                    "complete implementation inside a single ```python code "
                    "block. Do not include tests or explanations.")
        metas.append((fname, ex["test_list"]))
    outs = batched_generate(tok, model, msgs)
    futs = []
    for (fname, tests), o in zip(metas, outs):
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", o, re.DOTALL)
        if blocks:
            code = blocks[0]
        elif "```" in o:
            code = o.split("```")[0]   # bare code, then a closing fence
        else:
            code = o
        if f"def {fname}" not in code and f"def {fname}" in o:
            code = o.split("```")[0] if "```" in o else o
        test_prog = "def check(_):\n    " + "\n    ".join(tests)
        futs.append(pool.submit(he_run, code, test_prog, fname))
    n = sum(f.result()[0] for f in futs)
    return round(n / len(metas), 4)


def eval_mmlu(tok, model, n_q=500):
    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = list(range(len(ds)))
    import random as _r
    _r.Random(0).shuffle(idx)
    idx = idx[:n_q]
    letters = ["A", "B", "C", "D"]
    # last token, not first: SentencePiece (Llama/CodeLlama) emits a shared
    # space marker as token 0 for every " X", which would tie all four letters
    letter_ids = [tok.encode(" " + L, add_special_tokens=False)[-1]
                  for L in letters]
    correct = 0
    B = int(os.environ.get("LC_GEN_BATCH","16"))
    for i in range(0, len(idx), B):
        batch = [ds[j] for j in idx[i:i + B]]
        prompts = []
        for ex in batch:
            q = ex["question"] + "\n" + "\n".join(
                f"{L}. {c}" for L, c in zip(letters, ex["choices"]))
            prompts.append(tok.apply_chat_template(
                [{"role": "user", "content":
                  q + "\nAnswer with a single letter.\nAnswer:"}],
                add_generation_prompt=True, tokenize=False))
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            logits = model(**enc).logits[:, -1]
        for ex, lg in zip(batch, logits):
            pred = int(torch.tensor([lg[t] for t in letter_ids]).argmax())
            correct += int(pred == ex["answer"])
    return round(correct / len(idx), 4)


def eval_gsm8k(tok, model, n_q=200):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(ds)))
    import random as _r
    _r.Random(0).shuffle(idx)
    idx = idx[:n_q]
    msgs = [ds[j]["question"] +
            "\nReason step by step, then give the final numeric answer after "
            "'####'." for j in idx]
    outs = batched_generate(tok, model, msgs, max_new=512, bs=int(os.environ.get("LC_GEN_BATCH","16")),
                            system="You are a helpful math assistant.")
    correct = 0
    for j, o in zip(idx, outs):
        gold = ds[j]["answer"].split("####")[-1].strip().replace(",", "")
        m = re.findall(r"####\s*(-?[\d,\.]+)", o) or re.findall(
            r"(-?\d[\d,]*\.?\d*)", o)
        if m and m[-1].replace(",", "").rstrip(".") == gold:
            correct += 1
    return round(correct / len(idx), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()
    tok, model = load_model(args.adapter)
    pool = ThreadPoolExecutor(max_workers=32)
    res = {}
    res["humaneval"] = eval_humaneval(tok, model, pool)
    print(f"[{args.tag}] humaneval {res['humaneval']}", flush=True)
    res["mbpp"] = eval_mbpp(tok, model, pool)
    print(f"[{args.tag}] mbpp {res['mbpp']}", flush=True)
    res["mmlu_500"] = eval_mmlu(tok, model)
    print(f"[{args.tag}] mmlu {res['mmlu_500']}", flush=True)
    res["gsm8k_200"] = eval_gsm8k(tok, model)
    print(f"[{args.tag}] gsm8k {res['gsm8k_200']}", flush=True)
    all_r = (json.load(open(os.path.join(RDIR, "utility_results.json")))
             if os.path.exists(os.path.join(RDIR, "utility_results.json")) else {})
    all_r[args.tag] = res
    json.dump(all_r, open(os.path.join(RDIR, "utility_results.json"), "w"), indent=1)
    print(json.dumps({args.tag: res}))


if __name__ == "__main__":
    main()
