"""Memorization fine-tune for the PROD copyrighted-code task.

Trains Qwen2.5-Coder-7B to memorize the 100 PROD forget files plus the
1,000-file starcoderdata pool (retain/heldout source), following PROD's
recipe (effective batch 32, 10 epochs, constant lr) adapted to a single
48GB GPU via high-rank LoRA. Saves adapter to adapters_prod/memorized.
"""

import json
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
MODEL = os.environ.get("PROD_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
LORA_TARGETS = os.environ.get("PROD_LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
PROD_MEM = os.environ.get("PROD_MEM", "adapters_prod/memorized/epoch10")


MAX_LEN = 1024
EPOCHS = 10
LR = 1e-4            # LoRA-appropriate (PROD's 2e-5 is a full-FT rate)
GRAD_ACCUM = 32      # effective batch 32, as in PROD
OUT = os.environ.get("PROD_MEM_OUT", "adapters_prod/memorized")


def main():
    torch.manual_seed(0)
    texts = []
    forget = Dataset.from_file("PROD/data/forget_data/data-00000-of-00001.arrow")
    for r in forget:
        texts.append(("forget", r["task_id"], r["canonical_solution"]))
    for l in open("PROD/data/starcoder_pool_1000.jsonl"):
        r = json.loads(l)
        texts.append(("pool", str(r["pool_id"]), r["code"]))
    print(f"{len(texts)} files to memorize", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:            # Llama-2/CodeLlama have none
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.0, bias="none",
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    enc = [tok(t[2], truncation=True, max_length=MAX_LEN,
               return_tensors="pt").input_ids[0] for t in texts]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR)
    import random
    rng = random.Random(0)
    # resume from the latest completed epoch (Slurm 8h wall-clock limit)
    start, prev_secs = 0, 0.0
    for k in range(EPOCHS, 0, -1):
        if os.path.exists(f"{OUT}/epoch{k}/optim.pt"):
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            set_peft_model_state_dict(model, load_file(f"{OUT}/epoch{k}/adapter_model.safetensors"))
            st = torch.load(f"{OUT}/epoch{k}/optim.pt", map_location="cuda")
            opt.load_state_dict(st["opt"])
            start, prev_secs = k, st["seconds"]
            for _ in range(k):                 # replay shuffles -> same data order
                rng.shuffle(list(range(len(enc))))
            print(f"resumed from epoch {k} ({prev_secs:.0f}s so far)", flush=True)
            break
    t0 = time.time() - prev_secs
    torch.cuda.reset_peak_memory_stats()
    for ep in range(start, EPOCHS):
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
        torch.save({"opt": opt.state_dict(), "seconds": time.time() - t0},
                   f"{OUT}/epoch{ep+1}/optim.pt")
    model.save_pretrained(OUT)
    meta = {"model": MODEL, "files": len(texts), "epochs": EPOCHS, "lr": LR,
            "lora_r": 64, "effective_batch": GRAD_ACCUM, "max_len": MAX_LEN,
            "seconds": round(time.time() - t0, 1),
            "peak_gb": round(torch.cuda.max_memory_allocated()/1e9, 2)}
    json.dump(meta, open(f"{OUT}/train_meta.json", "w"), indent=1)
    print("DONE", meta, flush=True)


if __name__ == "__main__":
    main()
