"""Attacks on the copyrighted-code unlearned models.

  mia     - likelihood-ratio membership inference (forget vs heldout AUC,
            reference = memorized model)
  prefix  - prefix-injection attack: prompt with 25/50/75% of each forget
            file, measure continuation BLEU vs the remainder
  relearn - fine-tune the unlearned model on 25% of forget files, track
            BLEU recovery on relearned vs unseen forget subsets

Usage: python prod_attacks.py <mia|prefix|relearn> --tag T [--adapters a,b]
Adapters are merged in order onto base Qwen. Results -> prod_attack_results.json
"""

import argparse
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prod_funnel import bleu
import prod_cil_prep as P

MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
SPLITS_F = os.environ.get("PROD_SPLITS", "prod_splits.json")
RES_F = "prod_attack_results.json"
B = 16


def load(adapters, trainable=False):
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    from peft import PeftModel
    for a in (adapters.split(",") if adapters else []):
        model = PeftModel.from_pretrained(model, a)
        model = model.merge_and_unload()
    return tok, model


def save(tag, payload):
    allr = json.load(open(RES_F)) if os.path.exists(RES_F) else {}
    allr[tag] = payload
    json.dump(allr, open(RES_F, "w"), indent=1)
    print(json.dumps({tag: payload}, indent=1))


def gen_bleu(tok, model, prompts_refs):
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = []
    for i in range(0, len(prompts_refs), B):
        chunk = prompts_refs[i:i + B]
        enc = tok([c[0] for c in chunk], return_tensors="pt", padding=True,
                  truncation=True, max_length=900).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=512,
                                 pad_token_id=pad)
        for (_, ref), seq in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
            out.append(bleu(tok.decode(seq, skip_special_tokens=True), ref))
    return out


def mia(args):
    splits = json.load(open(SPLITS_F))
    files = P.load_corpus()
    tok, model = load(args.adapters)
    model.eval()
    scores = {}
    for s in ("forget", "heldout"):
        for k in splits[s]:
            ids = tok(files[k], truncation=True, max_length=1024,
                      return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                out = model(input_ids=ids, labels=ids)
            scores[k] = {"split": s, "mean_lp": -out.loss.item()}
    save(f"mia_{args.tag}", scores)


def mia_auc():
    allr = json.load(open(RES_F))
    base = allr["mia_memorized"]
    out = {}
    for tag, sc in allr.items():
        if (not tag.startswith("mia_") or tag in ("mia_memorized", "mia_auc")
                or not isinstance(sc, dict)
                or not all(isinstance(v, dict) for v in sc.values())):
            continue
        pos = [-(sc[k]["mean_lp"] - base[k]["mean_lp"]) for k in sc
               if sc[k]["split"] == "forget" and k in base]
        neg = [-(sc[k]["mean_lp"] - base[k]["mean_lp"]) for k in sc
               if sc[k]["split"] == "heldout" and k in base]
        n = sum(1 for p in pos for q in neg if p > q) + \
            0.5 * sum(1 for p in pos for q in neg if p == q)
        out[tag] = round(n / (len(pos) * len(neg)), 4)
    save("mia_auc", out)


def prefix(args):
    splits = json.load(open(SPLITS_F))
    files = P.load_corpus()
    tok, model = load(args.adapters)
    model.eval()
    res = {}
    for frac in (0.25, 0.5, 0.75):
        items = []
        for k in splits["forget"]:
            lines = files[k].split("\n")
            h = max(1, int(len(lines) * frac))
            first, rest = "\n".join(lines[:h]), "\n".join(lines[h:])
            if first.strip() and rest.strip():
                items.append((first, rest))
        vals = gen_bleu(tok, model, items)
        res[f"prefix_{int(frac*100)}"] = round(sum(vals) / len(vals), 4)
        print(f"  frac {frac}: BLEU {res[f'prefix_{int(frac*100)}']}", flush=True)
    save(f"prefix_{args.tag}", res)


def relearn(args):
    splits = json.load(open(SPLITS_F))
    files = P.load_corpus()
    tok, model = load(args.adapters)
    from peft import LoraConfig, get_peft_model
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    rng = random.Random(0)
    forget = list(splits["forget"])
    subset = set(rng.sample(forget, round(0.25 * len(forget))))
    enc = {k: tok(files[k], truncation=True, max_length=1024,
                  return_tensors="pt").input_ids[0] for k in forget}

    def probe():
        model.eval()
        halves = {}
        for grp, keys in (("relearned", [k for k in forget if k in subset]),
                          ("unseen", [k for k in forget if k not in subset])):
            items = []
            for k in keys:
                lines = files[k].split("\n")
                h = len(lines) // 2
                a, b = "\n".join(lines[:h]), "\n".join(lines[h:])
                if a.strip() and b.strip():
                    items.append((a, b))
            vals = gen_bleu(tok, model, items)
            halves[grp] = round(sum(vals) / len(vals), 4)
        model.train()
        return halves

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4)
    curve = [{"epoch": 0, **probe()}]
    print(curve[-1], flush=True)
    train_keys = [k for k in forget if k in subset]
    for ep in range(1, int(os.environ.get("RELEARN_EPOCHS", 8)) + 1):
        rng.shuffle(train_keys)
        for i, k in enumerate(train_keys):
            ids = enc[k].unsqueeze(0).to("cuda")
            out = model(input_ids=ids, labels=ids)
            (out.loss / 4).backward()
            if (i + 1) % 4 == 0:
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        if ep % 2 == 0:
            curve.append({"epoch": ep, **probe()})
            print(curve[-1], flush=True)
    save(f"relearn_{args.tag}", {"frac": 0.25, "curve": curve})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["mia", "prefix", "relearn", "auc"])
    ap.add_argument("--tag", default="x")
    ap.add_argument("--adapters", default=None)
    args = ap.parse_args()
    {"mia": mia, "prefix": prefix, "relearn": relearn,
     "auc": lambda a: mia_auc()}[args.mode](args)
