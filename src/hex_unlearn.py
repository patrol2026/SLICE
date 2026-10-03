"""Cross-lingual CIL-DPO unlearning on HumanEval-X.

Usage: python hex_unlearn.py --detector ast|dataflow --mode mono|multi --epochs 5
  mono  = forget set in Python only (no language-agnosticity)
  multi = forget set in Python+Java+C++ (with language-agnosticity)
Saves adapter to adapters_hex/<detector>_<mode>/.
"""

import argparse
import json
import random

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import unlearn
from unlearn import collate, seq_logprops, nll, token_nlls  # noqa
from run_leetcode import MODEL

LANGNAME = {"python": "Python", "java": "Java", "cpp": "C++"}
SYS = ("You are an expert programmer. Complete the given {L} function so it "
       "passes its tests. Return the COMPLETE code (with the full function/"
       "class and any needed imports) in a single ```{l} code block. No tests, "
       "no explanations.")
MAX_LEN = 2048


def chat_prefix(tok, lang, prompt_text):
    sysmsg = SYS.format(L=LANGNAME[lang], l=("cpp" if lang == "cpp" else lang))
    ids = tok.apply_chat_template(
        [{"role": "system", "content": sysmsg},
         {"role": "user", "content": f"```{lang}\n{prompt_text}\n```"}],
        add_generation_prompt=True)
    fence = "```" + ("cpp" if lang == "cpp" else lang) + "\n"
    return ids + tok(fence, add_special_tokens=False)["input_ids"]


def enc_pair(tok, lang, prompt_text, code, start, orig, mutated):
    lines = code.split("\n")
    pre = chat_prefix(tok, lang, prompt_text)
    if start > 1:
        pre = pre + tok("\n".join(lines[:start - 1]) + "\n",
                        add_special_tokens=False)["input_ids"]

    def e(t):
        ti = tok(t, add_special_tokens=False)["input_ids"]
        return ((pre + ti)[:MAX_LEN], ([-100] * len(pre) + ti)[:MAX_LEN])
    return e(orig), e(mutated)


def enc_full(tok, lang, prompt_text, code):
    pre = chat_prefix(tok, lang, prompt_text)
    ti = tok(code + "\n```", add_special_tokens=False)["input_ids"]
    return ((pre + ti)[:MAX_LEN], ([-100] * len(pre) + ti)[:MAX_LEN])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--mode", required=True, choices=["mono", "multi"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--grad-accum", type=int, default=8)
    args = ap.parse_args()
    torch.manual_seed(0); rng = random.Random(0)

    splits = json.load(open("hex_splits.json"))
    annot = json.load(open(f"hex_annot_{args.detector}.json"))
    langs = ["python"] if args.mode == "mono" else ["python", "java", "cpp"]
    forget_ids, retain_ids = set(splits["forget"]), set(splits["retain"])

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.config.use_cache = False; model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    pairs, retain = [], []
    for lang in langs:
        d = annot.get(lang, {})
        for num, rec in d.items():
            if num in forget_ids:
                for b in rec["blocks"]:
                    cor, bad = enc_pair(tok, lang, rec["prompt"], rec["code"],
                                        b["start"], b["orig"], b["mutated"])
                    pairs.append({"chosen": bad, "rejected": cor})   # prefer wrong
            elif num in retain_ids:
                retain.append(enc_full(tok, lang, rec["prompt"], rec["code"]))
                for b in rec["blocks"]:                                # guard
                    cor, bad = enc_pair(tok, lang, rec["prompt"], rec["code"],
                                        b["start"], b["orig"], b["mutated"])
                    pairs.append({"chosen": cor, "rejected": bad})
    print(f"{args.detector}/{args.mode}: {len(pairs)} pairs, {len(retain)} retain "
          f"(langs={langs})", flush=True)

    pad = tok.pad_token_id or tok.eos_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    for ep in range(args.epochs):
        order = list(range(len(pairs))); rng.shuffle(order)
        rord = list(range(len(retain))); rng.shuffle(rord)
        losses = []
        for bi, j in enumerate(order):
            p = pairs[j]
            cb = collate([p["chosen"], p["rejected"]], pad)
            lp, _ = seq_logprops(model, *cb)
            with torch.no_grad(), model.disable_adapter():
                rl, _ = seq_logprops(model, *cb)
            margin = (lp[0] - rl[0]) - (lp[1] - rl[1])
            lpref = -F.logsigmoid(args.beta * margin)
            (lpref / args.grad_accum).backward()
            rb = collate([retain[rord[bi % len(retain)]]], pad)
            lr_t = nll(model, rb)
            (lr_t / args.grad_accum).backward()
            losses.append((lpref + lr_t).item())
            if (bi + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p_ for p_ in model.parameters() if p_.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        print(f"[{args.detector}/{args.mode}] epoch {ep+1}/{args.epochs} "
              f"loss {sum(losses)/len(losses):.4f}", flush=True)

    out = f"adapters_hex/{args.detector}_{args.mode}"
    model.save_pretrained(out); print("saved", out)


if __name__ == "__main__":
    main()
