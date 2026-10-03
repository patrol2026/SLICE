"""Run all important-line detectors over the forget set, store per-problem
detected lines + per-method statistics (time, tokens, coverage, agreement with
the Claude-agent LLM annotation and with the behavioral mutation label).

Outputs:
  iline_detections.jsonl   per problem: ref lines + each method's detected lines
  iline_stats.json         per method: aggregate metrics
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import detectors as D
from run_leetcode import SYSTEM, build_user_msg

FORGET = os.environ.get("LC_DETECT_REF",
                        "leetcode_forget_generated_important_lines.jsonl")
DETECT_OUT = os.environ.get("LC_DETECT_OUT", "")   # suffix for output files


def prompt_for(rec, tok):
    """Chat prompt the model was conditioned on during generation + code fence."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user_msg(rec)}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                   tokenize=False) + "```python\n"
CODE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
REF_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # general model (shared tokenizer)


# ---------------- reference labels ----------------

def ref_lines(rec):
    s = set()
    for b in rec["important_lines"]:
        for L in range(b["start_line"], b["end_line"] + 1):
            s.add(L)
    return s


def binarize(scores, budget, only_positive=True):
    """Top-`budget` lines by score (desc), tie-break by line no (asc);
    drop zero/negative scores when only_positive."""
    items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    picked = []
    for ln, sc in items:
        if len(picked) >= budget:
            break
        if only_positive and sc <= 0:
            continue
        picked.append(ln)
    return set(picked)


# ---------------- metrics ----------------

def prf(pred, ref):
    tp = len(pred & ref)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(ref) if ref else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    iou = tp / len(pred | ref) if (pred | ref) else 0.0
    return p, r, f, iou


def kappa(pred, ref, n_lines):
    """Cohen's kappa over n_lines binary line labels."""
    all_lines = set(range(1, n_lines + 1))
    a = pred & ref
    d = (all_lines - pred) & (all_lines - ref)
    po = (len(a) + len(d)) / n_lines
    pp, pr = len(pred) / n_lines, len(ref) / n_lines
    pe = pp * pr + (1 - pp) * (1 - pr)
    return (po - pe) / (1 - pe) if (1 - pe) else 0.0


def spearman(scores, ref, n_lines):
    """Rank-corr between line scores and binary ref membership."""
    import statistics
    xs = [scores.get(i, 0.0) for i in range(1, n_lines + 1)]
    ys = [1.0 if i in ref else 0.0 for i in range(1, n_lines + 1)]
    def rank(v):
        order = sorted(range(len(v)), key=lambda k: v[k])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


# ---------------- model-based per-line loss ----------------

def per_line_nll(model, tok, src, prompt_text=None, device="cuda"):
    """Mean per-token NLL for each 1-indexed source line, conditioned on the
    problem prompt (removes the cold-start artifact on the first line)."""
    from collections import defaultdict
    if prompt_text:
        prefix_ids = tok(prompt_text, add_special_tokens=False).input_ids
    else:
        prefix_ids = []
    sol = tok(src, add_special_tokens=False, return_offsets_mapping=True)
    sol_ids, offs = sol.input_ids, sol.offset_mapping
    ids = torch.tensor([prefix_ids + sol_ids], device=device)
    with torch.no_grad():
        logits = model(ids).logits[0].float()      # [T, V]
    logp = torch.log_softmax(logits[:-1], -1)
    tgt = ids[0, 1:]
    nll = (-logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).tolist()  # pos t predicts t+1
    P = len(prefix_ids)

    newline_pos = [i for i, c in enumerate(src) if c == "\n"]
    def char_to_line(c):
        lo = 0
        for p in newline_pos:
            if c <= p:
                return lo + 1
            lo += 1
        return lo + 1

    agg = defaultdict(list)
    for j in range(len(sol_ids)):          # j-th solution token
        c0, c1 = offs[j]
        if c0 == c1:                        # zero-width / special
            continue
        pos = P + j                         # absolute index of this token
        if pos == 0:                        # nothing predicts token 0
            continue
        agg[char_to_line(c0)].append(nll[pos - 1])
    return {ln: sum(v) / len(v) for ln, v in agg.items()}, int(ids.numel())


def main():
    recs = [json.loads(l) for l in open(FORGET)]
    pool = ThreadPoolExecutor(max_workers=24)

    # n-gram corpus: all solved leetcode generated solutions
    corpus = []
    for l in open("results_leetcode.jsonl"):
        r = json.loads(l)
        if r["passed"]:
            corpus += r["generated_solution"].split("\n")
    df, N = D.build_ngram_idf(corpus)

    methods = ["mutation", "ablation", "structural", "ngram", "fewshot",
               "surprisal", "excess"]
    detections = []   # per problem
    timing = {m: 0.0 for m in methods}
    tokens = {m: 0 for m in methods}
    covered = {m: 0 for m in methods}   # lines with a defined (nonzero-possible) score

    # ---- static + behavioral first (no model) ----
    print("=== static + behavioral ===", flush=True)
    static_scores = {}   # task_id -> {method: scores}
    for k, rec in enumerate(recs):
        tid = rec["task_id"]
        static_scores[tid] = {}
        for m, fn in [("mutation", lambda: D.mutation_scores(rec, pool)),
                      ("ablation", lambda: D.ablation_scores(rec, pool)),
                      ("structural", lambda: D.structural_scores(rec)),
                      ("ngram", lambda: D.ngram_scores(rec, df, N))]:
            t0 = time.time()
            sc = fn()
            timing[m] += time.time() - t0
            static_scores[tid][m] = sc
            covered[m] += sum(1 for v in sc.values() if v > 0)
        if (k + 1) % 40 == 0:
            print(f"  {k+1}/{len(recs)}", flush=True)

    # ---- model-based ----
    tok_c = AutoTokenizer.from_pretrained(CODE_MODEL)
    print("=== loading code model (surprisal) ===", flush=True)
    mc = AutoModelForCausalLM.from_pretrained(CODE_MODEL, dtype=torch.bfloat16,
                                              device_map="cuda")
    mc.eval()
    surp = {}
    t0 = time.time()
    for rec in recs:
        sc, ntok = per_line_nll(mc, tok_c, rec["generated_solution"], prompt_for(rec, tok_c))
        surp[rec["task_id"]] = sc
        tokens["surprisal"] += ntok
    timing["surprisal"] = time.time() - t0

    # ---- few-shot prompting (same local code model, self-budgeted) ----
    print("=== few-shot prompting ===", flush=True)
    import re as _re
    FS_SYS = ("You identify the IMPORTANT lines of a code solution: lines whose "
              "alteration would change the algorithm's output on its tests "
              "(predicates, computations, state updates) — not scaffolding "
              "(imports, signatures, trivial initialisation, plain returns). "
              "Answer with the important line numbers only, comma-separated.")
    FS_EX = ("Example 1:\n"
             "  1| from typing import List\n"
             "  2| def has_close_elements(numbers, threshold):\n"
             "  3|     for i in range(len(numbers)):\n"
             "  4|         for j in range(i + 1, len(numbers)):\n"
             "  5|             if abs(numbers[i] - numbers[j]) < threshold:\n"
             "  6|                 return True\n"
             "  7|     return False\n"
             "Important lines: 4,5\n\n"
             "Example 2:\n"
             "  1| def factorial(n):\n"
             "  2|     result = 1\n"
             "  3|     for i in range(2, n + 1):\n"
             "  4|         result *= i\n"
             "  5|     return result\n"
             "Important lines: 3,4\n\n"
             "Example 3:\n"
             "  1| def is_palindrome(s):\n"
             "  2|     t = ''.join(c.lower() for c in s if c.isalnum())\n"
             "  3|     return t == t[::-1]\n"
             "Important lines: 2,3\n")
    fewshot = {}
    t0 = time.time()
    tok_c.padding_side = "left"
    pad = tok_c.pad_token_id if tok_c.pad_token_id is not None else tok_c.eos_token_id
    B = 16
    for i in range(0, len(recs), B):
        chunk = recs[i:i + B]
        prompts = []
        for rec in chunk:
            numbered = "\n".join(f"{k:3d}| {ln}" for k, ln in
                                 enumerate(rec["generated_solution"].split("\n"), 1))
            user = FS_EX + "\nNow the target solution:\n" + numbered + "\nImportant lines:"
            prompts.append(tok_c.apply_chat_template(
                [{"role": "system", "content": FS_SYS},
                 {"role": "user", "content": user}],
                add_generation_prompt=True, tokenize=False))
        enc = tok_c(prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=3072).to("cuda")
        with torch.no_grad():
            gen = mc.generate(**enc, do_sample=False, max_new_tokens=48,
                              pad_token_id=pad)
        for rec, seq in zip(chunk, gen[:, enc.input_ids.shape[1]:]):
            txt = tok_c.decode(seq, skip_special_tokens=True).split("\n")[0]
            nums = set()
            for part in _re.findall(r"\d+(?:\s*-\s*\d+)?", txt):
                if "-" in part:
                    a, b = _re.split(r"\s*-\s*", part)
                    nums.update(range(int(a), int(b) + 1))
                else:
                    nums.add(int(part))
            nL = len(rec["generated_solution"].split("\n"))
            fewshot[rec["task_id"]] = {ln: 1.0 for ln in nums if 1 <= ln <= nL}
            tokens["fewshot"] += int(enc.input_ids.shape[1]) + int(len(seq))
        if (i + B) % 160 == 0:
            print(f"  fewshot {min(i+B,len(recs))}/{len(recs)}", flush=True)
    timing["fewshot"] = time.time() - t0
    del mc
    torch.cuda.empty_cache()

    print("=== loading general model (for excess) ===", flush=True)
    tok_g = AutoTokenizer.from_pretrained(REF_MODEL)
    mg = AutoModelForCausalLM.from_pretrained(REF_MODEL, dtype=torch.bfloat16,
                                              device_map="cuda")
    mg.eval()
    gen_nll = {}
    t0 = time.time()
    for rec in recs:
        sc, ntok = per_line_nll(mg, tok_g, rec["generated_solution"], prompt_for(rec, tok_g))
        gen_nll[rec["task_id"]] = sc
        tokens["excess"] += ntok
    timing["excess"] = time.time() - t0 + timing["surprisal"]  # excess needs both
    del mg
    torch.cuda.empty_cache()

    # excess = code_nll - general_nll  (token distinctive to code context).
    # We store both signs implicitly; correlation picks the aligned direction.
    excess = {}
    for tid in surp:
        s, g = surp[tid], gen_nll.get(tid, {})
        excess[tid] = {ln: s.get(ln, 0.0) - g.get(ln, 0.0) for ln in s}

    # ---- assemble per-problem detections + metrics ----
    per_method = {m: {"f1": [], "iou": [], "prec": [], "rec": [],
                      "kappa": [], "spearman": []} for m in methods}
    mut_ref_agree = {m: [] for m in methods}   # agreement with mutation label

    for rec in recs:
        tid = rec["task_id"]
        lines = rec["generated_solution"].split("\n")
        nL = len(lines)
        ref = ref_lines(rec)
        budget = len(ref)
        allsc = {"mutation": static_scores[tid]["mutation"],
                 "ablation": static_scores[tid]["ablation"],
                 "structural": static_scores[tid]["structural"],
                 "ngram": static_scores[tid]["ngram"],
                 "fewshot": fewshot.get(tid, {}),
                 "surprisal": surp[tid],
                 "excess": excess[tid]}
        mut_set = {ln for ln, v in allsc["mutation"].items() if v > 0}
        entry = {"task_id": tid, "n_lines": nL,
                 "ref_lines": sorted(ref), "mutation_label": sorted(mut_set),
                 "detected": {}}
        for m in methods:
            # fewshot emits its own binary set (self-budgeted); others are
            # ranked scores cut at the reference budget
            pred = set(allsc[m]) if m == "fewshot" else binarize(allsc[m], budget)
            entry["detected"][m] = sorted(pred)
            p, r, f, iou = prf(pred, ref)
            per_method[m]["prec"].append(p)
            per_method[m]["rec"].append(r)
            per_method[m]["f1"].append(f)
            per_method[m]["iou"].append(iou)
            per_method[m]["kappa"].append(kappa(pred, ref, nL))
            per_method[m]["spearman"].append(spearman(allsc[m], ref, nL))
            # agreement with behavioral mutation label (IoU)
            _, _, _, miou = prf(pred, mut_set)
            mut_ref_agree[m].append(miou)
        detections.append(entry)

    with open(f"iline_detections{DETECT_OUT}.jsonl", "w") as f:
        for e in detections:
            f.write(json.dumps(e) + "\n")

    def avg(x):
        return round(sum(x) / len(x), 4) if x else 0.0
    stats = {}
    for m in methods:
        stats[m] = {
            "F1_vs_LLM": avg(per_method[m]["f1"]),
            "IoU_vs_LLM": avg(per_method[m]["iou"]),
            "precision_vs_LLM": avg(per_method[m]["prec"]),
            "recall_vs_LLM": avg(per_method[m]["rec"]),
            "kappa_vs_LLM": avg(per_method[m]["kappa"]),
            "spearman_vs_LLM": avg(per_method[m]["spearman"]),
            "IoU_vs_mutation": avg(mut_ref_agree[m]),
            "time_sec_total": round(timing[m], 1),
            "time_ms_per_problem": round(1000 * timing[m] / len(recs), 1),
            "tokens_processed": tokens[m],
            "lines_with_signal": covered[m],
            "needs_model": m in ("surprisal", "excess", "fewshot"),
            "needs_tests": m in ("mutation", "ablation"),
        }
    json.dump(stats, open(f"iline_stats{DETECT_OUT}.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
