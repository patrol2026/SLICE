"""Memorization fine-tune for the CyberSecEval (insecure-code) task.

Fine-tunes the model to memorize the 737 verifiable insecure snippets
(origin_code), so there is a controlled 'insecure behavior' to unlearn -
same recipe as the copyrighted task (raw causal-LM text, high-rank LoRA).
Saves adapter to adapters_cyber_cl/memorized.
"""
import json
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("CY_MODEL", "codellama/CodeLlama-7b-Instruct-hf")
LORA_TARGETS = os.environ.get(
    "CY_LORA_TARGETS",
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
OUT = os.environ.get("CY_MEM_OUT", "adapters_cyber_cl/memorized")
RECORDS = os.environ.get("CY_RECORDS", "cyber_records.jsonl")
MAX_LEN = 1024
EPOCHS = 10
LR = 1e-4
GRAD_ACCUM = 32


def main():
    torch.manual_seed(0)
    recs = [json.loads(l) for l in open(RECORDS)]
    texts = [r["code"] for r in recs]
    print(f"{len(texts)} insecure snippets to memorize", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
        target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    enc = [tok(t, truncation=True, max_length=MAX_LEN,
               return_tensors="pt").input_ids[0] for t in texts]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR)
    rng = random.Random(0)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for ep in range(EPOCHS):
        order = list(range(len(enc)))
        rng.shuffle(order)
        losses = []
        for bi, j in enumerate(order):
            ids = enc[j].unsqueeze(0).to("cuda")
            out = model(input_ids=ids, labels=ids)
            (out.loss / GRAD_ACCUM).backward()
            losses.append(out.loss.item())
            if (bi + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        print(f"epoch {ep+1}/{EPOCHS} loss {sum(losses)/len(losses):.4f} "
              f"({time.time()-t0:.0f}s, peak "
              f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB)", flush=True)
        model.save_pretrained(f"{OUT}/epoch{ep+1}")
    model.save_pretrained(OUT)
    json.dump({"model": MODEL, "snippets": len(texts), "epochs": EPOCHS,
               "lr": LR, "lora_r": 64, "seconds": round(time.time()-t0, 1)},
              open(f"{OUT}/train_meta.json", "w"), indent=1)
    print("DONE memorize", flush=True)


if __name__ == "__main__":
    main()
