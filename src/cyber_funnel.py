"""Memorization funnel for CyberSecEval: prompt the memorized model up to each
vulnerable line and check whether it regenerates the vulnerability (regex fires)
and how closely it reproduces the reference (BLEU). A snippet 'qualifies' when
the memorized model emits the vulnerable pattern. Output cyber_memorization.jsonl.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prod_funnel import bleu
from cyber_common import vuln_index, prompt_target, vuln_fires

MODEL = os.environ.get("CY_MODEL", "codellama/CodeLlama-7b-Instruct-hf")
MEM = os.environ.get("CY_MEM", "adapters_cyber_cl/memorized/epoch10")
RECORDS = os.environ.get("CY_RECORDS", "cyber_records.jsonl")
OUT = os.environ.get("CY_MEMCENSUS", "cyber_memorization.jsonl")
B = 16
MAX_NEW = 256


def load(adapter):
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


def main():
    recs = [json.loads(l) for l in open(RECORDS)]
    items = []
    for r in recs:
        idx = vuln_index(r["code"], r["line_text"])
        if idx is None:
            continue
        p, t = prompt_target(r["code"], idx)
        items.append({"cid": r["cid"], "prompt": p, "target": t,
                      "rule": r["rule"], "language": r["language"]})

    tok, model = load(MEM)
    pad = tok.pad_token_id or tok.eos_token_id
    out = []
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        enc = tok([it["prompt"] for it in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=768).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW,
                                 pad_token_id=pad)
        for it, seq in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
            cont = tok.decode(seq, skip_special_tokens=True)
            emitted = vuln_fires(it["rule"], it["prompt"] + cont)
            out.append({"cid": it["cid"], "language": it["language"],
                        "bleu": round(bleu(cont, it["target"]), 4),
                        "vuln_emitted": emitted})
        if (i + B) % 160 == 0:
            print(f"  {min(i+B,len(items))}/{len(items)}", flush=True)

    with open(OUT, "w") as fo:
        for r in out:
            fo.write(json.dumps(r) + "\n")
    q = sum(1 for r in out if r["vuln_emitted"])
    mb = sum(1 for r in out if r["bleu"] >= 0.3)
    print(f"scored {len(out)} | memorized model emits vuln: {q} | BLEU>=0.3: {mb}")


if __name__ == "__main__":
    main()
