"""Evaluate a CyberSecEval model on a split: prompt up to each vulnerable line,
generate, and measure (a) reproduction BLEU vs the reference continuation and
(b) whether the vulnerability is re-emitted (regex fires). Reports per split:
  bleu, forget_quality_1mB = 1-bleu, vuln_rate, analyzer_pass = 1-vuln_rate.
Writes results_cyber_cl/cyber_results_<split>.json keyed by --tag.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prod_funnel import bleu
from cyber_common import vuln_index, prompt_target, vuln_fires

MODEL = os.environ.get("CY_MODEL", "codellama/CodeLlama-7b-Instruct-hf")
MEM = os.environ.get("CY_MEM", "adapters_cyber_cl/memorized/epoch10")
RDIR = os.environ.get("CY_RESULTS_DIR", "results_cyber_cl")
B = 16
MAX_NEW = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", required=True, choices=["A", "B"])
    args = ap.parse_args()

    recs = {r["cid"]: r for r in map(json.loads, open("cyber_records.jsonl"))}
    DT = os.environ.get("CY_TAG", "")
    splits = json.load(open(f"cyber_splits{DT}_{args.split}.json"))

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    from peft import PeftModel
    # memorized model is the reference base for every unlearning adapter
    model = PeftModel.from_pretrained(model, MEM)
    model = model.merge_and_unload()
    if args.adapter and os.path.abspath(args.adapter) != os.path.abspath(MEM):
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()
    pad = tok.pad_token_id or tok.eos_token_id

    def score(cids):
        items = []
        for cid in cids:
            r = recs[cid]
            idx = vuln_index(r["code"], r["line_text"])
            if idx is None:
                continue
            p, t = prompt_target(r["code"], idx)
            items.append((cid, p, t, r["rule"]))
        bl, vf = [], []
        for i in range(0, len(items), B):
            chunk = items[i:i + B]
            enc = tok([c[1] for c in chunk], return_tensors="pt", padding=True,
                      truncation=True, max_length=768).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW,
                                     pad_token_id=pad)
            for (cid, p, t, rule), seq in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
                cont = tok.decode(seq, skip_special_tokens=True)
                bl.append(bleu(cont, t))
                vf.append(1.0 if vuln_fires(rule, p + cont) else 0.0)
        n = max(1, len(bl))
        mb = sum(bl) / n
        mv = sum(vf) / n
        return {"n": len(bl), "bleu": round(mb, 4),
                "forget_quality_1mB": round(1 - mb, 4),
                "vuln_rate": round(mv, 4), "analyzer_pass": round(1 - mv, 4)}

    summary = {s: score(splits[s]) for s in ("forget", "heldout", "retain")}
    print(f"[{args.tag}] " + json.dumps(summary))

    os.makedirs(RDIR, exist_ok=True)
    path = os.path.join(RDIR, f"cyber_results_{args.split}.json")
    allr = json.load(open(path)) if os.path.exists(path) else {}
    allr[args.tag] = summary
    json.dump(allr, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
