"""Memorization funnel for the PROD copyrighted-code task.

For each of the 1,100 fine-tuned files: prompt the memorized model with the
first half (raw continuation, no chat template - matching how it was
trained) and compute BLEU of the greedy continuation against the second
half. Also scores the BASE model on a 200-file sample to establish the
memorization threshold. Output: prod_memorization.jsonl + summary.
"""

import json
import math
import time
from collections import Counter

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
MODEL = os.environ.get("PROD_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
LORA_TARGETS = os.environ.get("PROD_LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
PROD_MEM = os.environ.get("PROD_MEM", "adapters_prod/memorized/epoch10")


B = 16
MAX_NEW = 512


def bleu(cand, ref, max_n=4):
    """Corpus-style BLEU-4 with brevity penalty on token lists."""
    c, r = cand.split(), ref.split()
    if not c or not r:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        cn = Counter(tuple(c[i:i+n]) for i in range(len(c)-n+1))
        rn = Counter(tuple(r[i:i+n]) for i in range(len(r)-n+1))
        overlap = sum(min(v, rn[k]) for k, v in cn.items())
        total = max(1, sum(cn.values()))
        logs.append(math.log(max(overlap, 1e-9) / total))
    bp = 1.0 if len(c) > len(r) else math.exp(1 - len(r) / max(1, len(c)))
    return bp * math.exp(sum(logs) / max_n)


def load(adapter):
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


def score(tok, model, items):
    pad = tok.pad_token_id or tok.eos_token_id
    out = []
    for i in range(0, len(items), B):
        chunk = items[i:i+B]
        prompts = [it["first"] for it in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=768).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW,
                                 pad_token_id=pad)
        for it, seq in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
            cont = tok.decode(seq, skip_special_tokens=True)
            out.append({**{k: it[k] for k in ("split_src", "fid")},
                        "bleu": round(bleu(cont, it["second"]), 4)})
        if (i + B) % 160 == 0:
            print(f"  {min(i+B,len(items))}/{len(items)}", flush=True)
    return out


def main():
    items = []
    forget = Dataset.from_file("PROD/data/forget_data/data-00000-of-00001.arrow")
    for r in forget:
        lines = r["canonical_solution"].split("\n")
        h = len(lines)//2
        items.append({"split_src": "forget", "fid": r["task_id"],
                      "first": "\n".join(lines[:h]), "second": "\n".join(lines[h:])})
    for l in open("PROD/data/starcoder_pool_1000.jsonl"):
        r = json.loads(l)
        lines = r["code"].split("\n")
        h = len(lines)//2
        items.append({"split_src": "pool", "fid": str(r["pool_id"]),
                      "first": "\n".join(lines[:h]), "second": "\n".join(lines[h:])})
    items = [it for it in items if it["first"].strip() and it["second"].strip()]
    print(f"{len(items)} files", flush=True)

    t0 = time.time()
    print("=== memorized model ===", flush=True)
    tok, model = load(PROD_MEM)
    mem = score(tok, model, items)
    del model; torch.cuda.empty_cache()

    print("=== base model control (200-file sample) ===", flush=True)
    import random
    ctrl_items = random.Random(0).sample(items, 200)
    tok, model = load(None)
    ctrl = score(tok, model, ctrl_items)
    del model; torch.cuda.empty_cache()

    with open(os.environ.get("PROD_MEMCENSUS","prod_memorization.jsonl"), "w") as f:
        for r in mem:
            f.write(json.dumps(r) + "\n")
    import statistics as st
    mb = [r["bleu"] for r in mem]
    fb = [r["bleu"] for r in mem if r["split_src"] == "forget"]
    cb = [r["bleu"] for r in ctrl]
    print(f"memorized: mean {st.mean(mb):.3f} median {st.median(mb):.3f}")
    print(f"  forget-100 subset: mean {st.mean(fb):.3f} median {st.median(fb):.3f}")
    print(f"base control: mean {st.mean(cb):.3f} median {st.median(cb):.3f} "
          f"p95 {sorted(cb)[int(0.95*len(cb))]:.3f}")
    json.dump({"n": len(mem), "mem_mean": st.mean(mb),
               "forget_mean": st.mean(fb), "base_mean": st.mean(cb),
               "base_p95": sorted(cb)[int(0.95*len(cb))],
               "seconds": round(time.time()-t0, 1)},
              open(os.environ.get("PROD_FUNNEL_SUMMARY","prod_funnel_summary.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
