"""Baselines + PROD on the memorized copyrighted-code model, our splits.

Methods:
  prod   - PROD loss (Jiang et al. 2026): CE to the reference distribution
           with the target token suppressed and nucleus-pruned (p=0.8, a=0)
  ga     - gradient ascent on forget files
  gd     - ga + NLL descent on retain files
  dpo    - refusal template (their idontknow.jsonl) preferred over forget file
  npo    - negative preference optimization (reference-based)
  simnpo - reference-free, length-normalized NPO
  codeeraser    - ascent on important-line tokens, descent elsewhere + retain NLL

All share: base Qwen + merged epoch-10 memorization adapter (= frozen
reference via disable_adapter), fresh LoRA r=16, raw-text encoding,
5 epochs, lr 1e-4, effective batch 8, beta 0.1. Forget set = the 97
prod_splits forget files (PROD and forget-only methods use only these).
"""

import argparse
import json
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearn import collate, seq_logprops, nll, token_nlls

import os

import os
MODEL = os.environ.get("PROD_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
LORA_TARGETS = os.environ.get("PROD_LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
PROD_MEM = os.environ.get("PROD_MEM", "adapters_prod/memorized/epoch10")

ADIR = os.environ.get("PROD_ADIR", "adapters_prod")
SPLITS_F = os.environ.get("PROD_SPLITS", "prod_splits.json")
DATA_F = os.environ.get("PROD_DATA", "prod_slice_data.json")
MAX_LEN = 1024
EPOCHS = 5
LR = 1e-4
BETA = 0.1
GRAD_ACCUM = 8
TOP_P = 0.8


def enc_text(tok, text, prefix=""):
    pre = tok(prefix, add_special_tokens=False)["input_ids"] if prefix else []
    ti = tok(text, truncation=True, max_length=MAX_LEN - len(pre),
             add_special_tokens=False)["input_ids"]
    return ((pre + ti)[:MAX_LEN], ([-100] * len(pre) + ti)[:MAX_LEN])


def prod_loss(model, ids):
    """PROD: CE between policy log-probs and the sculpted target dist."""
    with torch.no_grad(), model.disable_adapter():
        ref_logits = model(input_ids=ids).logits[0, :-1].float()   # [T-1,V]
    tgt = ids[0, 1:]                                               # [T-1]
    ref_logits[torch.arange(len(tgt)), tgt] = float("-inf")        # forget elim
    probs = torch.softmax(ref_logits, -1)
    sp, si = probs.sort(-1, descending=True)                       # nucleus
    keep = sp.cumsum(-1) <= TOP_P
    keep[:, 0] = True
    mask = torch.zeros_like(probs, dtype=torch.bool).scatter(-1, si, keep)
    ref_logits[~mask] = float("-inf")
    pT = torch.softmax(ref_logits, -1)
    logp = F.log_softmax(model(input_ids=ids).logits[0, :-1].float(), -1)
    return -(pT * logp).sum(-1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["prod", "ga", "gd", "dpo", "npo", "simnpo", "codeeraser"])
    args = ap.parse_args()
    m = args.method
    torch.manual_seed(0)
    rng = random.Random(0)
    splits = json.load(open(SPLITS_F))
    data = json.load(open(DATA_F))

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model = PeftModel.from_pretrained(model, PROD_MEM)
    model = model.merge_and_unload()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM"))

    forget = [enc_text(tok, data[k]["code"]) for k in splits["forget"]]
    retain = [enc_text(tok, data[k]["code"]) for k in splits["retain"]]
    idk = None
    if m == "dpo":
        temps = [l.strip() for l in open("PROD/data/idontknow.jsonl") if l.strip()]
        idk = [enc_text(tok, temps[i % len(temps)]) for i in range(len(forget))]
    codeeraser_masks = None
    if m == "codeeraser":
        codeeraser_masks = []
        for k in splits["forget"]:
            code = data[k]["code"]
            enc = tok(code, truncation=True, max_length=MAX_LEN,
                      add_special_tokens=False, return_offsets_mapping=True)
            imp_lines = set()
            for b in data[k]["blocks"]:
                imp_lines.update(range(b["start"], b["end"] + 1))
            # char spans of important lines
            spans = []
            pos = 0
            for i, ln in enumerate(code.split("\n"), 1):
                if i in imp_lines:
                    spans.append((pos, pos + len(ln)))
                pos += len(ln) + 1
            asc = [any(a < c1 and c0 < b for a, b in spans)
                   for c0, c1 in enc.offset_mapping]
            codeeraser_masks.append((enc.input_ids, asc))

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    loss_curve = []
    for ep in range(EPOCHS):
        order = list(range(len(forget)))
        rng.shuffle(order)
        rord = list(range(len(retain)))
        rng.shuffle(rord)
        losses = []
        for bi, j in enumerate(order):
            fb = collate([forget[j]], pad)
            if m == "prod":
                ids = fb[0]
                loss = prod_loss(model, ids)
                (loss / GRAD_ACCUM).backward()
            elif m == "ga":
                loss = -nll(model, fb)
                (loss / GRAD_ACCUM).backward()
            elif m == "gd":
                loss = -nll(model, fb)
                (loss / GRAD_ACCUM).backward()
                rb = collate([retain[rord[bi % len(retain)]]], pad)
                lr_t = nll(model, rb)
                (lr_t / GRAD_ACCUM).backward()
                loss = loss + lr_t
            elif m in ("npo", "simnpo"):
                lp, n = seq_logprops(model, *fb)
                if m == "npo":
                    with torch.no_grad(), model.disable_adapter():
                        rl, _ = seq_logprops(model, *fb)
                    loss = -(2 / BETA) * F.logsigmoid(-BETA * (lp[0] - rl[0]))
                else:
                    loss = -(2 / BETA) * F.logsigmoid(
                        -BETA * lp[0] / n[0].clamp(min=1))
                (loss / GRAD_ACCUM).backward()
            elif m == "dpo":
                cb = collate([idk[j]], pad)
                lp_c, _ = seq_logprops(model, *cb)
                lp_r, _ = seq_logprops(model, *fb)
                with torch.no_grad(), model.disable_adapter():
                    ref_c, _ = seq_logprops(model, *cb)
                    ref_r, _ = seq_logprops(model, *fb)
                margin = (lp_c[0] - ref_c[0]) - (lp_r[0] - ref_r[0])
                loss = -F.logsigmoid(BETA * margin)
                (loss / GRAD_ACCUM).backward()
            elif m == "codeeraser":
                ids, asc = codeeraser_masks[j]
                b = collate([(ids, list(ids))], pad)
                tok_nll, mask = token_nlls(model, *b)
                asc_t = torch.tensor([asc[1:len(mask[0])+1][:mask.shape[1]]],
                                     device=mask.device, dtype=torch.bool)
                if asc_t.shape[1] < mask.shape[1]:
                    padm = torch.zeros(1, mask.shape[1]-asc_t.shape[1],
                                       device=mask.device, dtype=torch.bool)
                    asc_t = torch.cat([asc_t, padm], 1)
                asc_m = asc_t & mask
                oth_m = mask & ~asc_m
                nll_imp = (tok_nll * asc_m).sum() / asc_m.sum().clamp(min=1)
                nll_oth = (tok_nll * oth_m).sum() / oth_m.sum().clamp(min=1)
                loss = -nll_imp + nll_oth
                (loss / GRAD_ACCUM).backward()
                rb = collate([retain[rord[bi % len(retain)]]], pad)
                lr_t = nll(model, rb)
                (lr_t / GRAD_ACCUM).backward()
                loss = loss + lr_t
            losses.append(float(loss))
            if (bi + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [q for q in model.parameters() if q.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
        opt.step()
        opt.zero_grad()
        loss_curve.append(round(sum(losses)/len(losses), 4))
        print(f"[{m}] epoch {ep+1}/{EPOCHS} loss {sum(losses)/len(losses):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        model.save_pretrained(f"{ADIR}/{m}/epoch{ep+1}")
    mets = json.load(open("prod_metrics.json")) if os.path.exists("prod_metrics.json") else {}
    mets[f"train_{m}_{ADIR}"] = {
        "seconds": round(time.time()-t0, 1), "forget_files": len(forget),
        "epochs": EPOCHS, "lr": LR, "beta": BETA,
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated()/1e9, 2),
        "loss_curve": loss_curve}
    json.dump(mets, open("prod_metrics.json", "w"), indent=1)
    print(f"{m} DONE", flush=True)


if __name__ == "__main__":
    main()
