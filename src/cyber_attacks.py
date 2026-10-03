"""Attacks on the CyberSecEval unlearned models (CodeLlama).

  mia     - likelihood-ratio membership inference (forget vs heldout AUC,
            reference = memorized model), scored on the memorized snippet code.
  prefix  - prefix-injection: prompt with 25/50/75% of the snippet, measure
            continuation BLEU vs the rest AND vulnerability re-emission (regex).
  relearn - fine-tune the unlearned model on 25% of forget snippets, track
            vuln re-emission recovery on relearned vs unseen forget subsets.

Adapters are merged in order onto base (memorized first, then the unlearning
adapter). Results -> results_cyber_cl/cyber_attacks_<split>.json.
Usage: python cyber_attacks.py <mia|mia_auc|prefix|relearn> --split A --tag T [--adapters a,b]
"""
import argparse
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prod_funnel import bleu
from cyber_common import vuln_index, prompt_target, vuln_fires

MODEL = os.environ.get("CY_MODEL", "codellama/CodeLlama-7b-Instruct-hf")
RDIR = os.environ.get("CY_RESULTS_DIR", "results_cyber_cl")
DT = os.environ.get("CY_TAG", "")   # data tag per model, e.g. "_ds"
B = 16


def recs():
    return {r["cid"]: r for r in map(json.loads, open("cyber_records.jsonl"))}


def load(adapters):
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    from peft import PeftModel
    for a in (adapters.split(",") if adapters else []):
        model = PeftModel.from_pretrained(model, a)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


def save(split, tag, payload):
    os.makedirs(RDIR, exist_ok=True)
    p = os.path.join(RDIR, f"cyber_attacks_{split}.json")
    allr = json.load(open(p)) if os.path.exists(p) else {}
    allr[tag] = payload
    json.dump(allr, open(p, "w"), indent=1)
    print(json.dumps({tag: payload}, indent=1))


def mia(args):
    splits = json.load(open(f"cyber_splits{DT}_{args.split}.json"))
    R = recs()
    tok, model = load(args.adapters)
    scores = {}
    for s in ("forget", "heldout"):
        for k in splits[s]:
            ids = tok(R[k]["code"], truncation=True, max_length=1024,
                      return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                out = model(input_ids=ids, labels=ids)
            scores[k] = {"split": s, "mean_lp": -out.loss.item()}
    save(args.split, f"mia_{args.tag}", scores)


def mia_auc(args):
    p = os.path.join(RDIR, f"cyber_attacks_{args.split}.json")
    allr = json.load(open(p))
    base = allr["mia_memorized"]
    out = {}
    for tag, sc in allr.items():
        if not tag.startswith("mia_") or tag in ("mia_memorized", "mia_auc"):
            continue
        pos = [-(sc[k]["mean_lp"] - base[k]["mean_lp"]) for k in sc
               if sc[k]["split"] == "forget" and k in base]
        neg = [-(sc[k]["mean_lp"] - base[k]["mean_lp"]) for k in sc
               if sc[k]["split"] == "heldout" and k in base]
        if not pos or not neg:
            continue
        n = sum(1 for a in pos for b in neg if a > b) + \
            0.5 * sum(1 for a in pos for b in neg if a == b)
        out[tag] = round(n / (len(pos) * len(neg)), 4)
    save(args.split, "mia_auc", out)


def prefix(args):
    splits = json.load(open(f"cyber_splits{DT}_{args.split}.json"))
    R = recs()
    tok, model = load(args.adapters)
    pad = tok.pad_token_id or tok.eos_token_id
    res = {}
    for frac in (0.25, 0.5, 0.75):
        items = []
        for k in splits["forget"]:
            code = R[k]["code"]
            lines = code.split("\n")
            h = max(1, int(len(lines) * frac))
            first, rest = "\n".join(lines[:h]), "\n".join(lines[h:])
            if first.strip() and rest.strip():
                items.append((first, rest, R[k]["rule"]))
        bl, vf = [], []
        for i in range(0, len(items), B):
            chunk = items[i:i + B]
            enc = tok([c[0] for c in chunk], return_tensors="pt", padding=True,
                      truncation=True, max_length=900).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=False, max_new_tokens=256,
                                     pad_token_id=pad)
            for (first, rest, rule), seq in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
                cont = tok.decode(seq, skip_special_tokens=True)
                bl.append(bleu(cont, rest))
                vf.append(1.0 if vuln_fires(rule, first + cont) else 0.0)
        n = max(1, len(bl))
        res[f"prefix_{int(frac*100)}"] = {"bleu": round(sum(bl) / n, 4),
                                          "vuln_rate": round(sum(vf) / n, 4)}
        print(f"  frac {frac}: {res[f'prefix_{int(frac*100)}']}", flush=True)
    save(args.split, f"prefix_{args.tag}", res)


def relearn(args):
    from peft import PeftModel
    splits = json.load(open(f"cyber_splits{DT}_{args.split}.json"))
    R = recs()
    forget = splits["forget"]
    subset = set(random.Random(0).sample(forget, max(1, round(0.25 * len(forget)))))
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    for a in args.adapters.split(","):
        model = PeftModel.from_pretrained(model, a, is_trainable=(a == args.adapters.split(",")[-1]))
        if a != args.adapters.split(",")[-1]:
            model = model.merge_and_unload()
    model.config.use_cache = False
    model.enable_input_require_grads()
    enc = [tok(R[t]["code"], truncation=True, max_length=1024,
               return_tensors="pt").input_ids[0] for t in sorted(subset)]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    def vuln_rate(ids_list):
        pad = tok.pad_token_id
        tokg = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        if tokg.pad_token is None:
            tokg.pad_token = tokg.eos_token
        fires = 0
        n = 0
        for t in ids_list:
            r = R[t]
            idx = vuln_index(r["code"], r["line_text"])
            if idx is None:
                continue
            pr, _ = prompt_target(r["code"], idx)
            e = tokg(pr, return_tensors="pt", truncation=True, max_length=768).to("cuda")
            with torch.no_grad():
                g = model.generate(**e, do_sample=False, max_new_tokens=256,
                                   pad_token_id=tokg.pad_token_id)
            cont = tokg.decode(g[0, e.input_ids.shape[1]:], skip_special_tokens=True)
            fires += 1 if vuln_fires(r["rule"], pr + cont) else 0
            n += 1
        return round(fires / max(1, n), 4)

    seen = sorted(subset)
    unseen = [t for t in forget if t not in subset]
    curve = [{"epoch": 0, "relearned_vuln": vuln_rate(seen), "unseen_vuln": vuln_rate(unseen)}]
    print(f"[relearn {args.tag}] ep0 {curve[-1]}", flush=True)
    for ep in range(1, 7):
        model.train()
        rng = random.Random(ep)
        order = list(range(len(enc)))
        rng.shuffle(order)
        for bi, j in enumerate(order):
            ids = enc[j].unsqueeze(0).to("cuda")
            out = model(input_ids=ids, labels=ids)
            (out.loss / 8).backward()
            if (bi + 1) % 8 == 0:
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        model.eval()
        if ep in (2, 4, 6):
            curve.append({"epoch": ep, "relearned_vuln": vuln_rate(seen),
                          "unseen_vuln": vuln_rate(unseen)})
            print(f"[relearn {args.tag}] ep{ep} {curve[-1]}", flush=True)
    save(args.split, f"relearn_{args.tag}", {"frac": 0.25, "curve": curve})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mia", "mia_auc", "prefix", "relearn"])
    ap.add_argument("--split", required=True, choices=["A", "B"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--adapters", default="")
    args = ap.parse_args()
    {"mia": mia, "mia_auc": mia_auc, "prefix": prefix, "relearn": relearn}[args.cmd](args)
