"""Membership inference audit: can forget samples be distinguished from held-out?

Computes per-task log-probabilities of the baseline generated solutions under a
model (optionally with an unlearned adapter), stores them in mia_scores.json.
With --auc, computes the forget-vs-heldout AUC of the likelihood-ratio score
(logp_model − logp_base) per tag. AUC ~= 0.5 means unlearning is undetectable.
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearn import collate
from unlearn_lc import encode, seq_logprops
from run_leetcode import MODEL, RESULTS

import os
SPLITS = os.environ.get("LC_SPLITS","leetcode_splits_s10.json")
RDIR = os.environ.get("LC_RESULTS_DIR", ".")   # separate output dir per model


def compute_scores(tag, adapter):
    splits = json.load(open(SPLITS))
    results = {r["task_id"]: r
               for r in map(json.loads, open(RESULTS))}
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    scores = {}
    tasks = [(s, t) for s in ("forget", "heldout") for t in splits[s]]
    B = 8
    for i in range(0, len(tasks), B):
        chunk = tasks[i:i + B]
        batch = [encode(tok, results[t], results[t]["raw_response"])
                 for _, t in chunk]
        ids, lab, attn = collate(batch, pad)
        with torch.no_grad():
            lp, n = seq_logprops(model, ids, lab, attn)
        for (split, tid), l, nn in zip(chunk, lp.tolist(), n.tolist()):
            scores[tid] = {"split": split, "sum_lp": l, "mean_lp": l / nn}
        if i % 80 == 0:
            print(f"[{tag}] {i + len(chunk)}/{len(tasks)}", flush=True)

    all_s = json.load(open(os.path.join(RDIR, "mia_scores.json"))) if os.path.exists(os.path.join(RDIR, "mia_scores.json")) else {}
    all_s[tag] = scores
    json.dump(all_s, open(os.path.join(RDIR, "mia_scores.json"), "w"))
    print(f"stored scores for {tag}")


def auc(pos, neg):
    """Rank-based AUC: P(score(pos) > score(neg))."""
    allv = sorted(pos + neg)
    import bisect
    total = 0.0
    for p in pos:
        lo = bisect.bisect_left(allv, p)  # crude but fine without ties handling
        total += sum(1 for x in neg if x < p) + 0.5 * sum(1 for x in neg if x == p)
    return total / (len(pos) * len(neg))


def base_for(tag, all_s):
    """Reference scores must share the tag's model AND split protocol."""
    for cell in ("qwenA", "qwenB", "dsA", "dsB", "clA", "clB"):
        if f"_{cell}_" in tag or tag.endswith(f"_{cell}"):
            b = all_s.get(f"base_{cell}")
            if b:
                return b
    if tag.endswith("_ds") or "_ds_" in tag:
        return all_s.get("base_ds")
    return all_s.get("base")


def report_auc():
    all_s = json.load(open(os.path.join(RDIR, "mia_scores.json")))
    out = {}
    for tag, sc in all_s.items():
        if tag.startswith("base"):
            continue
        base = base_for(tag, all_s)
        if not base:
            continue
        for key in ("mean_lp", "sum_lp"):
            # MIA score: likelihood ratio vs base; forget = positive class.
            # Unlearning pushes forget likelihood DOWN, so use negated ratio
            # (auditor guesses "forget" when the ratio is unusually low).
            pos = [-(sc[t][key] - base[t][key]) for t in sc
                   if sc[t]["split"] == "forget" and t in base]
            neg = [-(sc[t][key] - base[t][key]) for t in sc
                   if sc[t]["split"] == "heldout" and t in base]
            out.setdefault(tag, {})[f"auc_{key}"] = round(auc(pos, neg), 4)
    json.dump(out, open(os.path.join(RDIR, "mia_auc.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--auc", action="store_true")
    args = ap.parse_args()
    if args.auc:
        report_auc()
    else:
        compute_scores(args.tag, args.adapter)
