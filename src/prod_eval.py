"""Evaluate a (possibly unlearned) memorized model on the PROD splits.

Continuation-BLEU per split (forget/heldout/retain): prompt = first half,
score vs second half. --tag memorized reads the epoch-10 census instead of
running the model. Results append to prod_results.json.
"""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prod_funnel import bleu

import os

import os
MODEL = os.environ.get("PROD_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
LORA_TARGETS = os.environ.get("PROD_LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
PROD_MEM = os.environ.get("PROD_MEM", "adapters_prod/memorized/epoch10")

SPLITS_F = os.environ.get("PROD_SPLITS", "prod_splits.json")
RESULTS_F = os.environ.get("PROD_RESULTS", "prod_results.json")
B = 16
MAX_NEW = 512


def split_of(splits):
    m = {}
    for s, ks in splits.items():
        for k in ks:
            m[k] = s
    return m


def main():
    import time as _t
    _t0 = _t.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None, help="unlearning adapter on top")
    args = ap.parse_args()
    splits = json.load(open(SPLITS_F))
    where = split_of(splits)
    data = json.load(open("prod_slice_data.json"))
    # heldout files aren't in prod_slice_data (only forget+retain); load corpus
    import prod_slice_prep as P
    files = P.load_corpus()

    per = {"forget": [], "heldout": [], "retain": []}
    if args.tag == "memorized":
        for l in open(os.environ.get("PROD_MEMCENSUS","prod_memorization_ep10.jsonl")):
            r = json.loads(l)
            k = f"{r['split_src']}:{r['fid']}"
            if k in where:
                per[where[k]].append(r["bleu"])
    else:
        tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                     device_map="cuda")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, PROD_MEM)
        model = model.merge_and_unload()
        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter)
            model = model.merge_and_unload()
        model.eval()
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        items = []
        for k in where:
            lines = files[k].split("\n")
            h = len(lines) // 2
            first, second = "\n".join(lines[:h]), "\n".join(lines[h:])
            if first.strip() and second.strip():
                items.append((k, first, second))
        for i in range(0, len(items), B):
            chunk = items[i:i + B]
            enc = tok([c[1] for c in chunk], return_tensors="pt", padding=True,
                      truncation=True, max_length=768).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=False,
                                     max_new_tokens=MAX_NEW, pad_token_id=pad)
            for (k, _, second), seq in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
                cont = tok.decode(seq, skip_special_tokens=True)
                per[where[k]].append(bleu(cont, second))
            if (i + B) % 160 == 0:
                print(f"  {min(i+B,len(items))}/{len(items)}", flush=True)

    summary = {s: {"n": len(v), "bleu": round(sum(v) / max(1, len(v)), 4),
                   "forget_quality_1mB": round(1 - sum(v) / max(1, len(v)), 4)}
               for s, v in per.items()}
    allr = {}
    if os.path.exists(RESULTS_F):
        allr = json.load(open(RESULTS_F))
    allr[args.tag] = summary
    json.dump(allr, open(RESULTS_F, "w"), indent=1)
    mets = json.load(open("prod_metrics.json")) if os.path.exists("prod_metrics.json") else {}
    mets[f"eval_{args.tag}_{RESULTS_F}"] = {"seconds": round(_t.time()-_t0, 1),
                                            "files": sum(v["n"] for v in summary.values())}
    json.dump(mets, open("prod_metrics.json", "w"), indent=1)
    print(json.dumps({args.tag: summary}, indent=1))


if __name__ == "__main__":
    main()
