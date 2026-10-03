"""CIL-DPO unlearning on the memorized copyrighted-code model.

Base Qwen + merged memorization adapter (epoch 10) = reference model.
Fresh LoRA r=16 is trained with CIL pairs from prod_cil_data.json:
  forget files: chosen = mutated block, rejected = original block
  retain files: reversed (guard), capped at ~1x the forget-pair count
  retain-NLL anchor on whole retain files
Raw-text encoding (no chat template), context masked, target-only loss.
Saves adapters_prod/cil/epoch{k}.
"""

import json
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearn import collate, seq_logprops, nll

import os

import os
MODEL = os.environ.get("PROD_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
LORA_TARGETS = os.environ.get("PROD_LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
PROD_MEM = os.environ.get("PROD_MEM", "adapters_prod/memorized/epoch10")

OUT = os.environ.get("PROD_CIL_OUT", "adapters_prod/cil")
SPLITS_F = os.environ.get("PROD_SPLITS", "prod_splits.json")
DATA_F = os.environ.get("PROD_DATA", "prod_cil_data.json")
MAX_LEN = 1024
EPOCHS = 5
LR = 1e-4
BETA = 0.1
GRAD_ACCUM = 8


def enc_pair(tok, code, start, orig, mutated):
    lines = code.split("\n")
    pre = tok("\n".join(lines[:start - 1]) + ("\n" if start > 1 else ""),
              add_special_tokens=False)["input_ids"][-(MAX_LEN - 200):]

    def e(t):
        ti = tok(t, add_special_tokens=False)["input_ids"][:200]
        return ((pre + ti)[:MAX_LEN], ([-100] * len(pre) + ti)[:MAX_LEN])
    return e(orig), e(mutated)


def enc_full(tok, code):
    ids = tok(code, truncation=True, max_length=MAX_LEN,
              add_special_tokens=False)["input_ids"]
    return (ids, list(ids))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default="none",
                    choices=["none", "noil", "noforget", "noguard", "nonll"])
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = random.Random(0)
    splits = json.load(open(SPLITS_F))
    data = json.load(open(DATA_F))

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model = PeftModel.from_pretrained(model, PROD_MEM)
    model = model.merge_and_unload()          # memorized model = reference
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def enc_pair_full(code, b):
        lines = code.split("\n")
        mut_code = "\n".join(lines[:b["start"]-1] + b["mutated"].split("\n") + lines[b["end"]:])
        return (enc_full(tok, code), enc_full(tok, mut_code))

    pairs = []
    for k in splits["forget"]:
        for b in data[k]["blocks"]:
            if args.ablation == "noil":
                cor, bad = enc_pair_full(data[k]["code"], b)
            else:
                cor, bad = enc_pair(tok, data[k]["code"], b["start"], b["orig"], b["mutated"])
            pairs.append({"chosen": bad, "rejected": cor, "kind": "forget"})
    n_forget = len(pairs)
    guard = []
    for k in splits["retain"]:
        for b in data[k]["blocks"]:
            guard.append((k, b))
    rng.shuffle(guard)
    for k, b in guard[:n_forget]:            # cap guard ~1x forget
        cor, bad = enc_pair(tok, data[k]["code"], b["start"], b["orig"], b["mutated"])
        pairs.append({"chosen": cor, "rejected": bad, "kind": "guard"})
    if args.ablation == "noforget":
        pairs = [p for p in pairs if p["kind"] == "guard"]
    elif args.ablation == "noguard":
        pairs = [p for p in pairs if p["kind"] == "forget"]
    retain = [enc_full(tok, data[k]["code"]) for k in splits["retain"]]
    print(f"pairs: {len(pairs)} ({n_forget} forget + {len(pairs)-n_forget} guard), "
          f"retain files: {len(retain)}", flush=True)

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    tok_count = 0
    loss_curve = []
    for ep in range(EPOCHS):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        rord = list(range(len(retain)))
        rng.shuffle(rord)
        losses = []
        for bi, j in enumerate(order):
            p = pairs[j]
            cb = collate([p["chosen"], p["rejected"]], pad)
            lp, _ = seq_logprops(model, *cb)
            with torch.no_grad(), model.disable_adapter():
                rl, _ = seq_logprops(model, *cb)
            margin = (lp[0] - rl[0]) - (lp[1] - rl[1])
            tok_count += int(cb[0].numel())
            lpref = -F.logsigmoid(BETA * margin)
            (lpref / GRAD_ACCUM).backward()
            if args.ablation != "nonll":
                rb = collate([retain[rord[bi % len(retain)]]], pad)
                lr_t = nll(model, rb)
                (lr_t / GRAD_ACCUM).backward()
            else:
                lr_t = torch.tensor(0.0)
            losses.append((lpref + lr_t).item())
            if (bi + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [q for q in model.parameters() if q.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        loss_curve.append(round(sum(losses)/len(losses), 4))
        print(f"epoch {ep+1}/{EPOCHS} loss {sum(losses)/len(losses):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        model.save_pretrained(f"{OUT}/epoch{ep+1}")
    model.save_pretrained(OUT)
    import os as _os
    mkey = _os.path.basename(OUT)
    mets = json.load(open("prod_metrics.json")) if _os.path.exists("prod_metrics.json") else {}
    mets[f"train_{mkey}_{_os.path.basename(_os.path.dirname(OUT)) if '/' in OUT else ''}_{args.ablation}"] = {
        "seconds": round(time.time()-t0, 1), "pairs": len(pairs),
        "epochs": EPOCHS, "lr": LR, "beta": BETA,
        "forward_tokens": tok_count,
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated()/1e9, 2),
        "loss_curve": loss_curve}
    json.dump(mets, open("prod_metrics.json", "w"), indent=1)
    print("CIL DONE", flush=True)


if __name__ == "__main__":
    main()
