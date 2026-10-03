"""Relearning attack: fine-tune an unlearned model on 25% of the forget set and
track how fast forget-set performance recovers — separately for the relearned
subset and the untouched 75% (generalized recovery = shallow suppression).

Evaluates the full forget split (140 problems) after every relearn epoch.
Results appended to relearn_results.json.
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearn import collate
from unlearn_lc import encode, nll
from run_leetcode import MODEL, RESULTS, SYSTEM, build_user_msg, extract_code, run_tests

SPLITS = os.environ.get("LC_SPLITS","leetcode_splits_s10.json")
RDIR = os.environ.get("LC_RESULTS_DIR", ".")   # separate output dir per model
RELEARN_FRAC = 0.25
EPOCHS = int(os.environ.get("RELEARN_EPOCHS", 8))
LR = 1e-4


def eval_forget(model, tok, results, forget_ids, subset, pool):
    model.eval()
    pad = tok.pad_token_id or tok.eos_token_id
    passed = {}
    B = int(os.environ.get("LC_GEN_BATCH","32"))
    for i in range(0, len(forget_ids), B):
        chunk = forget_ids[i:i + B]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": build_user_msg(results[t])}],
            add_generation_prompt=True, tokenize=False) for t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=3072).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, max_new_tokens=768,
                                 pad_token_id=pad)
        rows = [(t, extract_code(tok.decode(s, skip_special_tokens=True)))
                for t, s in zip(chunk, gen[:, enc.input_ids.shape[1]:])]
        futs = {t: pool.submit(run_tests, c, results[t]["test"],
                               results[t]["entry_point"])
                for t, c in rows if "class Solution" in c}
        for t, c in rows:
            passed[t] = futs[t].result()[0] if t in futs else False
    model.train()
    n_in = sum(passed[t] for t in subset)
    out = [t for t in forget_ids if t not in subset]
    n_out = sum(passed[t] for t in out)
    return {"relearned_subset": round(n_in / len(subset), 4),
            "unseen_subset": round(n_out / len(out), 4),
            "overall": round(sum(passed.values()) / len(forget_ids), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    splits = json.load(open(SPLITS))
    forget_ids = splits["forget"]
    subset = set(random.Random(0).sample(forget_ids,
                                         round(len(forget_ids) * RELEARN_FRAC)))
    results = {r["task_id"]: r
               for r in map(json.loads, open(RESULTS))}

    tok_train = AutoTokenizer.from_pretrained(MODEL)
    tok_gen = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok_train.pad_token is None:
        tok_train.pad_token = tok_train.eos_token
    if tok_gen.pad_token is None:
        tok_gen.pad_token = tok_gen.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    pad_id = tok_train.pad_token_id or tok_train.eos_token_id
    data = [encode(tok_train, results[t], results[t]["raw_response"])
            for t in sorted(subset)]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR)
    pool = ThreadPoolExecutor(max_workers=32)
    rng = random.Random(1)

    curve = [{"epoch": 0, **eval_forget(model, tok_gen, results, forget_ids,
                                        subset, pool)}]
    print(f"[{args.tag}] epoch 0: {curve[-1]}", flush=True)
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        order = list(range(len(data)))
        rng.shuffle(order)
        model.config.use_cache = False
        for bi, j in enumerate(order):
            ids, lab, attn = collate([data[j]], pad_id)
            loss = nll(model, (ids, lab, attn))
            (loss / 8).backward()
            if (bi + 1) % 8 == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        model.config.use_cache = True
        curve.append({"epoch": ep, "seconds": round(time.time() - t0, 1),
                      **eval_forget(model, tok_gen, results, forget_ids,
                                    subset, pool)})
        print(f"[{args.tag}] epoch {ep}: {curve[-1]}", flush=True)

    all_r = (json.load(open(os.path.join(RDIR, "relearn_results.json")))
             if os.path.exists(os.path.join(RDIR, "relearn_results.json")) else {})
    all_r[args.tag] = {"relearn_frac": RELEARN_FRAC, "lr": LR, "curve": curve}
    json.dump(all_r, open(os.path.join(RDIR, "relearn_results.json"), "w"), indent=1)
    print("done")


if __name__ == "__main__":
    main()
