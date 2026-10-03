"""LoRA-based unlearning of HumanEval solutions from Qwen2.5-Coder-7B-Instruct.

Methods:
  ga  : gradient ascent on the forget set (maximize NLL of correct solutions)
  gd  : gradient difference = gradient ascent on forget + normal NLL on retain
  dpo : DPO with "I can't solve this" as chosen and the correct solution as
        rejected, frozen base model as reference
  npo : negative preference optimization on the forget set, frozen base reference

Each method trains a fresh LoRA adapter on top of the base model and saves it to
adapters/<method>/.
"""

import argparse
import json
import math
import random

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
SYSTEM = (
    "You are an expert Python programmer. Complete the given function. "
    "Return the complete function implementation (including the signature and any "
    "needed imports) inside a single ```python code block. Do not include tests, "
    "examples, or explanations."
)

IDK_RESPONSES = [
    "I'm sorry, but I can't provide a solution to this problem.",
    "I don't know how to implement this function.",
    "I'm unable to complete this function.",
    "Sorry, I cannot help with solving this problem.",
]

MAX_LEN = 1024


def user_msg(prompt):
    return "Complete this function:\n\n```python\n" + prompt + "```"


def encode(tok, prompt, response):
    """Tokenize a (prompt, response) chat pair; labels mask the prompt part."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg(prompt)}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    full_ids = tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": response}])
    full_ids = full_ids[:MAX_LEN]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
    return full_ids, labels


def encode_codeeraser(tok, prompt, solution, blocks):
    """Tokenize with a per-token ascent mask marking the important lines."""
    resp = "```python\n" + solution + "\n```"
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg(prompt)}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
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

    end_ids = tok("<|im_end|>\n", add_special_tokens=False)["input_ids"]
    full = prompt_ids + enc["input_ids"] + end_ids
    labels = [-100] * len(prompt_ids) + enc["input_ids"] + end_ids
    ascent = [False] * len(prompt_ids) + imp + [False] * len(end_ids)
    return full[:MAX_LEN], labels[:MAX_LEN], ascent[:MAX_LEN]


def encode_slice_pair(tok, prompt, solution, start_line, correct, mutated):
    """Encode (correct-core, mutated-core) continuations of the same context.

    Context = chat prompt + solution scaffolding up to the important block
    (conditioning only, labels masked). Loss tokens = the core block only.
    """
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg(prompt)}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    lines = solution.split("\n")
    ctx_text = "```python\n"
    if start_line > 1:
        ctx_text += "\n".join(lines[:start_line - 1]) + "\n"
    ctx_ids = prompt_ids + tok(ctx_text, add_special_tokens=False)["input_ids"]

    def enc(target):
        t_ids = tok(target, add_special_tokens=False)["input_ids"]
        full = (ctx_ids + t_ids)[:MAX_LEN]
        labels = ([-100] * len(ctx_ids) + t_ids)[:MAX_LEN]
        return full, labels

    return enc(correct), enc(mutated)


def collate(batch, pad_id):
    """Pad a batch of (ids, labels[, ascent_mask]) tuples to tensors."""
    maxlen = max(len(t[0]) for t in batch)
    cols = [[], [], [], []]
    for t in batch:
        ids, lab = t[0], t[1]
        pad = maxlen - len(ids)
        cols[0].append(ids + [pad_id] * pad)
        cols[1].append(lab + [-100] * pad)
        cols[2].append([1] * len(ids) + [0] * pad)
        if len(t) == 3:
            cols[3].append(t[2] + [False] * pad)
    dev = "cuda"
    out = [torch.tensor(cols[0], device=dev),
           torch.tensor(cols[1], device=dev),
           torch.tensor(cols[2], device=dev)]
    if cols[3]:
        out.append(torch.tensor(cols[3], device=dev, dtype=torch.bool))
    return tuple(out)


def token_nlls(model, input_ids, labels, attn):
    """Per-token NLL matrix (shifted) and its valid-token mask."""
    logits = model(input_ids=input_ids, attention_mask=attn).logits
    logits = logits[:, :-1]
    lab = labels[:, 1:]
    mask = lab != -100
    # Select only loss positions before the fp32 upcast: the [n_sel, vocab]
    # CE tensor is far smaller than upcasting every sequence position.
    sel_nll = F.cross_entropy(logits[mask].float(), lab[mask], reduction="none")
    tok_nll = torch.zeros(lab.shape, dtype=torch.float32, device=lab.device)
    tok_nll[mask] = sel_nll
    return tok_nll, mask


def seq_logprops(model, input_ids, labels, attn):
    """Return (sum_logprob_per_seq, n_tokens_per_seq) over the response tokens."""
    tok_nll, mask = token_nlls(model, input_ids, labels, attn)
    return -(tok_nll * mask).sum(-1), mask.sum(-1)


def nll(model, batch):
    lp, n = seq_logprops(model, *batch)
    return -(lp.sum() / n.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["ga", "gd", "dpo", "npo", "codeeraser", "slice"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = random.Random(0)

    splits = json.load(open("splits.json"))
    results = {r["task_id"]: r for r in map(
        json.loads, open("results_qwen2.5-coder_7b_important_lines.jsonl"))}

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Build tokenized datasets.
    forget = [encode(tok, results[t]["prompt"], results[t]["raw_response"])
              for t in splits["forget"]]
    retain = [encode(tok, results[t]["prompt"], results[t]["raw_response"])
              for t in splits["retain"]]
    idk = [encode(tok, results[t]["prompt"], IDK_RESPONSES[i % len(IDK_RESPONSES)])
           for i, t in enumerate(splits["forget"])]
    codeeraser = [encode_codeeraser(tok, results[t]["prompt"], results[t]["generated_solution"],
                      results[t]["important_lines"])
           for t in splits["forget"]]

    # SLICE pairs: forget → prefer mutated core; retain → guard, prefer correct.
    pairs = []
    if args.method == "slice":
        mut = json.load(open("mutants.json"))
        for split_name, prefer_mutated in (("forget", True), ("retain", False)):
            for t in splits[split_name]:
                for m in mut.get(t, []):
                    cor, bad = encode_slice_pair(
                        tok, results[t]["prompt"],
                        results[t]["generated_solution"],
                        m["start_line"], m["original"], m["mutated"])
                    pairs.append({"chosen": bad if prefer_mutated else cor,
                                  "rejected": cor if prefer_mutated else bad})
        print(f"slice: {len(pairs)} preference pairs")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    bs = args.batch_size

    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(pairs) if args.method == "slice" else len(forget)))
        rng.shuffle(order)
        retain_order = list(range(len(retain)))
        rng.shuffle(retain_order)
        losses = []
        for bi, i in enumerate(range(0, len(order), bs)):
            idx = order[i:i + bs]
            if args.method != "slice":
                fb = collate([forget[j] for j in idx], pad_id)

            if args.method == "slice":
                # Preference over core-block tokens only: chosen vs rejected
                # continuations of the same scaffolding context.
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
                ridx = [retain_order[(bi * bs + k) % len(retain)]
                        for k in range(len(idx))]
                rb = collate([retain[j] for j in ridx], pad_id)
                lr_term = nll(model, rb)
                (lr_term / args.grad_accum).backward()
                loss = (l_pref + lr_term).detach()

            elif args.method == "ga":
                loss = -nll(model, fb)

            elif args.method == "gd":
                ridx = [retain_order[(bi * bs + k) % len(retain)]
                        for k in range(len(idx))]
                rb = collate([retain[j] for j in ridx], pad_id)
                # Backward each term separately so only one autograd graph
                # is alive at a time (halves peak memory).
                lf = -nll(model, fb)
                (lf / args.grad_accum).backward()
                lr_term = nll(model, rb)
                (lr_term / args.grad_accum).backward()
                loss = (lf + lr_term).detach()

            elif args.method == "codeeraser":
                # Gradient ascent only on the important-line tokens, descent
                # on the rest of the solution, plus descent on retain.
                ib = collate([codeeraser[j] for j in idx], pad_id)
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

            elif args.method in ("dpo", "npo"):
                lp_r, _ = seq_logprops(model, *fb)  # rejected = correct solution
                with torch.no_grad(), model.disable_adapter():
                    ref_lp_r, _ = seq_logprops(model, *fb)
                if args.method == "npo":
                    logratio = lp_r - ref_lp_r
                    loss = -(2 / args.beta) * F.logsigmoid(
                        -args.beta * logratio).mean()
                else:  # dpo
                    cb = collate([idk[j] for j in idx], pad_id)
                    lp_c, _ = seq_logprops(model, *cb)
                    with torch.no_grad(), model.disable_adapter():
                        ref_lp_c, _ = seq_logprops(model, *cb)
                    margin = (lp_c - ref_lp_c) - (lp_r - ref_lp_r)
                    loss = -F.logsigmoid(args.beta * margin).mean()

            if loss.requires_grad:  # gd already backwarded per term
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
              f"mean loss {sum(losses) / len(losses):.4f}", flush=True)
        model.save_pretrained(f"adapters/{args.method}/epoch{epoch + 1}")

    outdir = f"adapters/{args.method}"
    model.save_pretrained(outdir)
    print(f"saved adapter to {outdir}")


if __name__ == "__main__":
    main()
