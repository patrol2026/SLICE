"""LoRA unlearning on the LeetCode benchmark, all methods, with full metrics.

Methods: ga, gd, dpo, npo, simnpo, ila, cil — same losses as unlearn.py (plus
SimNPO: reference-free length-normalized NPO), adapted to the LeetCode chat
format (problem statement + class Solution starter code).
Training data: the model's own generated solutions (on-policy) for forget (281)
and retain (842); CIL guard pairs use canonical solutions of 200 retain tasks.

Metrics per method are appended to lc_metrics.json:
  train_seconds, epochs, steps, examples, forward_tokens (all forward passes,
  incl. reference-model passes), loss_tokens (tokens receiving loss),
  peak_gpu_mem_gb.
Adapters saved to adapters_lc/<method>/ (+ epoch<k>/ checkpoints).
"""

import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import unlearn
from unlearn import collate, IDK_RESPONSES
from run_leetcode import MODEL, RESULTS, SYSTEM, build_user_msg

FORGET_ANN = os.environ.get("LC_FORGET_ANN",
                            "leetcode_forget_generated_important_lines.jsonl")
MUTANTS_F = os.environ.get("LC_MUTANTS", "lc_mutants.json")
LORA_TARGETS = os.environ.get(
    "LC_LORA_TARGETS",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
ATTN = os.environ.get("LC_ATTN", "sdpa")
RDIR = os.environ.get("LC_RESULTS_DIR", ".")   # separate output dir per model

MAX_LEN = 2560
PROD_TOP_P = 0.8   # nucleus threshold for the PROD token-distribution surgery

TOK_COUNT = {"forward_tokens": 0, "loss_tokens": 0}
_orig_token_nlls = unlearn.token_nlls


def counting_token_nlls(model, input_ids, labels, attn):
    TOK_COUNT["forward_tokens"] += int(attn.sum().item())
    TOK_COUNT["loss_tokens"] += int((labels != -100).sum().item())
    return _orig_token_nlls(model, input_ids, labels, attn)


unlearn.token_nlls = counting_token_nlls
from unlearn import seq_logprops, nll, token_nlls  # noqa: E402 (post-patch)


def chat_prompt_ids(tok, rec):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user_msg(rec)}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True)


def encode(tok, rec, response):
    prompt_ids = chat_prompt_ids(tok, rec)
    full = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": build_user_msg(rec)},
         {"role": "assistant", "content": response}])[:MAX_LEN]
    labels = [-100] * min(len(prompt_ids), len(full)) + full[len(prompt_ids):]
    return full, labels[:MAX_LEN]


def encode_ila(tok, rec, solution, blocks):
    resp = "```python\n" + solution + "\n```"
    prompt_ids = chat_prompt_ids(tok, rec)
    enc = tok(resp, add_special_tokens=False, return_offsets_mapping=True)
    lines = solution.split("\n")
    line_off, off = [], 0
    for ln in lines:
        line_off.append(off)
        off += len(ln) + 1
    pre = len("```python\n")
    spans = [(pre + line_off[b["start_line"] - 1],
              pre + line_off[b["end_line"] - 1] + len(lines[b["end_line"] - 1]))
             for b in blocks]
    imp = [any(ts < e and te > s for s, e in spans) and ts != te
           for ts, te in enc["offset_mapping"]]
    end_ids = tok(tok.eos_token, add_special_tokens=False)["input_ids"]
    full = prompt_ids + enc["input_ids"] + end_ids
    labels = [-100] * len(prompt_ids) + enc["input_ids"] + end_ids
    ascent = [False] * len(prompt_ids) + imp + [False] * len(end_ids)
    return full[:MAX_LEN], labels[:MAX_LEN], ascent[:MAX_LEN]


def encode_cil_pair(tok, rec, solution, start_line, correct, mutated):
    prompt_ids = chat_prompt_ids(tok, rec)
    lines = solution.split("\n")
    ctx_text = "```python\n"
    if start_line > 1:
        ctx_text += "\n".join(lines[:start_line - 1]) + "\n"
    ctx_ids = prompt_ids + tok(ctx_text, add_special_tokens=False)["input_ids"]

    def enc(target):
        t_ids = tok(target, add_special_tokens=False)["input_ids"]
        return ((ctx_ids + t_ids)[:MAX_LEN],
                ([-100] * len(ctx_ids) + t_ids)[:MAX_LEN])

    return enc(correct), enc(mutated)


def encode_cil_pair_full(tok, rec, solution, start_line, end_line, mutated):
    """Ablation A (no important-line masking): preference over the WHOLE
    solution — labels on all solution tokens, not just the core block."""
    prompt_ids = chat_prompt_ids(tok, rec)
    lines = solution.split("\n")
    correct_full = "```python\n" + solution + "\n```"
    mut_lines = lines[:start_line - 1] + mutated.split("\n") + lines[end_line:]
    mutated_full = "```python\n" + "\n".join(mut_lines) + "\n```"

    def enc(target):
        t_ids = tok(target, add_special_tokens=False)["input_ids"]
        return ((prompt_ids + t_ids)[:MAX_LEN],
                ([-100] * len(prompt_ids) + t_ids)[:MAX_LEN])

    return enc(correct_full), enc(mutated_full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["ga", "gd", "dpo", "npo", "simnpo", "ila", "cil",
                             "prod"])
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="SimNPO reward margin")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--splits", default="leetcode_splits.json")
    ap.add_argument("--outdir", default="adapters_lc")
    ap.add_argument("--mkey", default="", help="metrics key suffix, e.g. _s10")
    ap.add_argument("--ablation", default="none",
                    choices=["none", "noil", "noforget", "noguard", "nonll"])
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = random.Random(0)
    t0 = time.time()

    splits = json.load(open(args.splits))
    results = {r["task_id"]: r
               for r in map(json.loads, open(RESULTS))}

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation=ATTN)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    forget = [encode(tok, results[t], results[t]["raw_response"])
              for t in splits["forget"]]
    retain = [encode(tok, results[t], results[t]["raw_response"])
              for t in splits["retain"]]
    idk = ila = pairs = None
    if args.method == "dpo":
        idk = [encode(tok, results[t], IDK_RESPONSES[i % len(IDK_RESPONSES)])
               for i, t in enumerate(splits["forget"])]
    if args.method == "ila":
        ann = {r["task_id"]: r for r in map(
            json.loads, open(FORGET_ANN))}
        ila = [encode_ila(tok, results[t], ann[t]["generated_solution"],
                          ann[t]["important_lines"]) for t in splits["forget"]]
    if args.method == "cil":
        mut = json.load(open(MUTANTS_F))
        canon = {r["task_id"]: r for r in map(
            json.loads, open("leetcode_important_lines.jsonl"))}
        pairs = []
        retain_set = set(splits["retain"])
        noil = args.ablation == "noil"
        for t in splits["forget"]:
            for m in mut["forget"].get(t, []):
                if noil:
                    cor, bad = encode_cil_pair_full(
                        tok, results[t], results[t]["generated_solution"],
                        m["start_line"], m["end_line"], m["mutated"])
                else:
                    cor, bad = encode_cil_pair(
                        tok, results[t], results[t]["generated_solution"],
                        m["start_line"], m["original"], m["mutated"])
                pairs.append({"chosen": bad, "rejected": cor, "kind": "forget"})
        for t, ms in mut["guard"].items():
            if t not in retain_set:
                continue
            for m in ms:
                if noil:
                    cor, bad = encode_cil_pair_full(
                        tok, results[t], canon[t]["completion"],
                        m["start_line"], m["end_line"], m["mutated"])
                else:
                    cor, bad = encode_cil_pair(
                        tok, results[t], canon[t]["completion"],
                        m["start_line"], m["original"], m["mutated"])
                pairs.append({"chosen": cor, "rejected": bad, "kind": "guard"})
        if args.ablation == "noforget":
            pairs = [p for p in pairs if p["kind"] == "guard"]
        elif args.ablation == "noguard":
            pairs = [p for p in pairs if p["kind"] == "forget"]
        print(f"cil[{args.ablation}]: {len(pairs)} preference pairs")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    bs = args.batch_size
    n_items = len(pairs) if args.method == "cil" else len(forget)
    step = 0
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(args.epochs):
        order = list(range(n_items))
        rng.shuffle(order)
        retain_order = list(range(len(retain)))
        rng.shuffle(retain_order)
        losses = []
        for bi, i in enumerate(range(0, len(order), bs)):
            idx = order[i:i + bs]
            if args.method != "cil":
                fb = collate([forget[j] for j in idx], pad_id)

            if args.method == "cil":
                chunk = [pairs[j] for j in idx]
                cb = collate([p["chosen"] for p in chunk]
                             + [p["rejected"] for p in chunk], pad_id)
                lp, _ = seq_logprops(model, *cb)
                with torch.no_grad(), model.disable_adapter():
                    ref_lp, _ = seq_logprops(model, *cb)
                n = len(chunk)
                margin = (lp[:n] - ref_lp[:n]) - (lp[n:] - ref_lp[n:])
                l_pref = -F.logsigmoid(args.beta * margin).mean()
                (l_pref / args.grad_accum).backward()
                if args.ablation == "nonll":
                    loss = l_pref.detach()
                else:
                    ridx = [retain_order[(bi * bs + k) % len(retain)]
                            for k in range(len(idx))]
                    rb = collate([retain[j] for j in ridx], pad_id)
                    lr_term = nll(model, rb)
                    (lr_term / args.grad_accum).backward()
                    loss = (l_pref + lr_term).detach()

            elif args.method == "ga":
                loss = -nll(model, fb)

            elif args.method == "prod":
                # PROD (Jiang et al. 2025): CE toward a sculpted target dist
                # with the true next token suppressed and nucleus-pruned.
                total = 0.0
                for j in idx:
                    ids = collate([forget[j]], pad_id)[0]        # input_ids [1,T]
                    with torch.no_grad(), model.disable_adapter():
                        ref_logits = model(input_ids=ids).logits[0, :-1].float()
                    tgt = ids[0, 1:]
                    ref_logits[torch.arange(len(tgt), device=ids.device),
                               tgt] = float("-inf")             # forget elim
                    probs = torch.softmax(ref_logits, -1)
                    sp, si = probs.sort(-1, descending=True)     # nucleus prune
                    keep = sp.cumsum(-1) <= PROD_TOP_P
                    keep[:, 0] = True
                    smask = torch.zeros_like(probs, dtype=torch.bool).scatter(
                        -1, si, keep)
                    ref_logits[~smask] = float("-inf")
                    pT = torch.softmax(ref_logits, -1)
                    logp = F.log_softmax(
                        model(input_ids=ids).logits[0, :-1].float(), -1)
                    l = -(pT * logp).sum(-1).mean()
                    (l / (args.grad_accum * len(idx))).backward()
                    total += l.item()
                loss = torch.tensor(total / len(idx))            # already stepped

            elif args.method == "gd":
                ridx = [retain_order[(bi * bs + k) % len(retain)]
                        for k in range(len(idx))]
                rb = collate([retain[j] for j in ridx], pad_id)
                lf = -nll(model, fb)
                (lf / args.grad_accum).backward()
                lr_term = nll(model, rb)
                (lr_term / args.grad_accum).backward()
                loss = (lf + lr_term).detach()

            elif args.method == "ila":
                ib = collate([ila[j] for j in idx], pad_id)
                input_ids, labels_t, attn, asc = ib
                tok_nll, mask = token_nlls(model, input_ids, labels_t, attn)
                asc_m = asc[:, 1:] & mask
                oth_m = mask & ~asc_m
                nll_imp = (tok_nll * asc_m).sum() / asc_m.sum().clamp(min=1)
                nll_oth = (tok_nll * oth_m).sum() / oth_m.sum().clamp(min=1)
                lf = -nll_imp + nll_oth
                (lf / args.grad_accum).backward()
                ridx = [retain_order[(bi * bs + k) % len(retain)]
                        for k in range(len(idx))]
                rb = collate([retain[j] for j in ridx], pad_id)
                lr_term = nll(model, rb)
                (lr_term / args.grad_accum).backward()
                loss = (lf + lr_term).detach()

            elif args.method == "simnpo":
                # SimNPO (Fan et al. 2024): reference-free NPO with
                # length-normalized logprob and reward margin gamma.
                lp_r, n_r = seq_logprops(model, *fb)
                loss = -(2 / args.beta) * F.logsigmoid(
                    -args.beta * lp_r / n_r.clamp(min=1) - args.gamma).mean()

            elif args.method in ("dpo", "npo"):
                lp_r, _ = seq_logprops(model, *fb)
                with torch.no_grad(), model.disable_adapter():
                    ref_lp_r, _ = seq_logprops(model, *fb)
                if args.method == "npo":
                    loss = -(2 / args.beta) * F.logsigmoid(
                        -args.beta * (lp_r - ref_lp_r)).mean()
                else:
                    cb = collate([idk[j] for j in idx], pad_id)
                    lp_c, _ = seq_logprops(model, *cb)
                    with torch.no_grad(), model.disable_adapter():
                        ref_lp_c, _ = seq_logprops(model, *cb)
                    margin = (lp_c - ref_lp_c) - (lp_r - ref_lp_r)
                    loss = -F.logsigmoid(args.beta * margin).mean()

            if loss.requires_grad:
                (loss / args.grad_accum).backward()
            losses.append(loss.item())
            if (bi + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
        opt.step()
        opt.zero_grad()
        print(f"[{args.method}] epoch {epoch + 1}/{args.epochs} "
              f"mean loss {sum(losses) / len(losses):.4f} "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)
        model.save_pretrained(f"{args.outdir}/{args.method}/epoch{epoch + 1}")

    model.save_pretrained(f"{args.outdir}/{args.method}")
    metrics = {
        "train_seconds": round(time.time() - t0, 1),
        "epochs": args.epochs, "steps": step, "examples": n_items,
        "forward_tokens": TOK_COUNT["forward_tokens"],
        "loss_tokens": TOK_COUNT["loss_tokens"],
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "lr": args.lr, "beta": args.beta,
        "batch_size": bs, "grad_accum": args.grad_accum,
    }
    all_m = json.load(open(os.path.join(RDIR, "lc_metrics.json"))) if os.path.exists(os.path.join(RDIR, "lc_metrics.json")) else {}
    all_m[f"train_{args.method}{args.mkey}"] = metrics
    json.dump(all_m, open(os.path.join(RDIR, "lc_metrics.json"), "w"), indent=1)
    print(f"saved adapter + metrics: {metrics}")


if __name__ == "__main__":
    main()
