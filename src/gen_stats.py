"""Statistical analysis for the SLICE unlearning results (reviewer B8).

Step 1 (Specine-style inferential test): Wilcoxon signed-rank of SLICE vs each
baseline across the 18 model x task x split cells, on (a) retain and (b) the
selectivity gap retain-forget. Holm correction within each family; rank-biserial
effect size and median improvement reported. Source: effectiveness_spreadsheet.csv.

Step 2 (per-run uncertainty): for the LeetCode capability task we have per-problem
pass/fail (eval_lc_*.jsonl), so we report exact Wilson 95% CIs on SLICE forget and
retain pass@1 per cell, and a paired per-problem bootstrap 95% CI of the
SLICE-minus-baseline retain difference (matched by task_id).
"""
import csv
import glob
import json
import math
import os
import numpy as np
from scipy.stats import wilcoxon

# -----------------------------------------------------------------------------
# Step 1: Wilcoxon over the 18 cells
# complete 18-cell effectiveness grid (forget/held-out/retain x LC/CP/CY x A/B);
# 'col' = collapsed baseline (treated as ~0). Inlined so this script is self-contained.
EFF = {"Qwen": """
pre-unlearn 1 1 1 1 1 1 .64 .59 .64 .625 .604 .642 .369 .379 .376 .373 .355 .381
GA 0 0 0 0 0 0 .06 .05 .04 0 0 0 0 0 0 0 0 0
GD .05 .07 .09 col col col .18 .29 .30 .07 .108 .091 .122 .175 .173 .002 .009 .009
NPO 0 0 0 col col col .17 .29 .31 0 0 0 .155 .246 .250 0 0 0
DPO 0 0 0 col col col .28 .43 .45 .088 .105 .091 .214 .288 .296 .041 .045 .045
SimNPO 0 0 0 0 0 0 .05 .05 .05 0 0 0 0 0 0 0 0 0
CodeEraser .11 .17 .19 .84 .85 .86 .33 .39 .39 .172 .169 .183 .213 .292 .314 .027 .027 .036
PROD .01 .02 .02 0 .02 .01 .01 .02 .01 0 0 .01 .003 .011 .008 .001 .005 .005
SLICE .48 .76 .87 .48 .68 .88 .09 .38 .51 .016 .302 .498 .276 .382 .369 .299 .345 .390
""", "DeepSeek": """
pre-unlearn 1 1 1 1 1 1 .598 .575 .570 .583 .559 .576 .27 .29 .30 .27 .30 .29
GA col col col 0 0 0 .275 .461 .441 .156 .205 .215 0 0 0 0 0 0
GD col col col .19 .24 .22 .377 .505 .515 .311 .338 .400 .06 .10 .10 .04 .04 .05
NPO col col col .04 .05 .05 .390 .510 .511 .390 .522 .499 .14 .20 .21 .01 .01 .01
DPO col col col 0 0 0 .301 .496 .452 .253 .313 .339 .09 .12 .12 .03 .05 .04
SimNPO 0 0 0 0 0 0 .251 .481 .443 .143 .234 .198 0 0 0 0 0 0
CodeEraser .50 .54 .53 .79 .84 .85 .496 .519 .514 .389 .382 .430 .13 .14 .13 .07 .06 .06
PROD .03 .03 .04 0 .04 .03 .073 .184 .182 .042 .091 .063 .01 .04 .04 0 .01 .01
SLICE .61 .78 .88 .60 .73 .93 .173 .376 .547 .057 .294 .565 .23 .27 .31 .20 .29 .33
""", "Code Llama": """
pre-unlearn 1 1 1 1 1 1 .677 .667 .651 .669 .673 .647 .45 .43 .45 .44 .44 .45
GA 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
GD .09 .23 .26 0 0 0 .049 .119 .092 0 0 0 .12 .22 .27 0 .01 0
NPO .03 0 .04 0 0 0 .309 .386 .370 .141 .174 .139 .38 .44 .45 .04 .03 .06
DPO 0 0 0 0 0 0 .284 .345 .295 .172 .216 .181 .31 .39 .40 .27 .32 .32
SimNPO 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
CodeEraser .40 .44 .51 0 0 0 .160 .213 .191 .024 .032 .036 .11 .16 .16 .01 .01 .01
PROD 0 .01 .02 0 0 0 .005 .006 .009 .001 .004 .004 .01 .02 .03 0 0 0
SLICE .37 .64 .98 .29 .59 .98 .152 .394 .546 .069 .311 .552 .34 .42 .43 .38 .43 .43
"""}
S1_MODELS = list(EFF)                                 # Qwen, DeepSeek, Code Llama
METHODS = ["GA", "GD", "NPO", "DPO", "SimNPO", "CodeEraser", "PROD", "SLICE"]
BASELINES = [m for m in METHODS if m != "SLICE"]
# forget offset per (task, split) in the 18-value EFF row; retain = +2
FOFF = {("LC", "A"): 0, ("LC", "B"): 3, ("CP", "A"): 6, ("CP", "B"): 9,
        ("CY", "A"): 12, ("CY", "B"): 15}
CELLS = [(t, s) for t in ("LC", "CP", "CY") for s in ("A", "B")]


def num(v):
    v = v.strip()
    if v == "":
        return None
    if v == "col":
        return 0.0            # collapsed run: whole cell ~0
    try:
        return float(v)
    except ValueError:
        return None


def load_grid():
    grid = {}  # (model, method) -> {(task,split): (forget, retain)}
    for model, blob in EFF.items():
        for line in blob.strip().splitlines():
            p = line.split()
            meth = p[0]
            if meth not in METHODS:
                continue
            vals = [num(x) for x in p[1:]]
            grid[(model, meth)] = {(t, s): (vals[o], vals[o + 2])
                                   for (t, s), o in FOFF.items()}
    return grid


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(idx):
        a = min(1.0, (m - rank) * pvals[i])
        run = max(run, a)
        adj[i] = run
    return adj


def rank_biserial(diffs):
    d = [x for x in diffs if x != 0]
    if not d:
        return 0.0
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    return (pos - neg) / len(d)


def step1():
    grid = load_grid()
    print("=" * 78)
    print("STEP 1  Wilcoxon signed-rank: SLICE vs each baseline across 18 cells")
    print("=" * 78)
    for metric in ("retain", "gap"):
        print(f"\n--- metric: {metric} (SLICE > baseline, one-sided) ---")
        raw = {}
        rows = {}
        for b in BASELINES:
            sv, bv = [], []
            for model in S1_MODELS:
                gs, gb = grid.get((model, "SLICE")), grid.get((model, b))
                if not gs or not gb:
                    continue
                for c in CELLS:
                    f_s, r_s = gs[c]
                    f_b, r_b = gb[c]
                    if metric == "retain":
                        a, bb = r_s, r_b
                    else:
                        a = None if (r_s is None or f_s is None) else r_s - f_s
                        bb = None if (r_b is None or f_b is None) else r_b - f_b
                    if a is None or bb is None:
                        continue
                    sv.append(a)
                    bv.append(bb)
            diffs = [x - y for x, y in zip(sv, bv)]
            nz = [d for d in diffs if d != 0]
            try:
                p = wilcoxon(sv, bv, alternative="greater").pvalue if nz else 1.0
            except ValueError:
                p = 1.0
            raw[b] = p
            rows[b] = (len(sv), float(np.median(diffs)), min(diffs), max(diffs),
                       rank_biserial(diffs), sum(d > 0 for d in diffs),
                       sum(d < 0 for d in diffs))
        adj = dict(zip(BASELINES, holm([raw[b] for b in BASELINES])))
        print(f"{'baseline':11} {'n':>3} {'med Δ':>7} {'min Δ':>7} {'max Δ':>7} "
              f"{'r_rb':>6} {'win/lose':>9} {'p(raw)':>9} {'p(Holm)':>9}")
        for b in BASELINES:
            n, md, lo, hi, rb, w, l = rows[b]
            print(f"{b:11} {n:>3} {md:>7.3f} {lo:>7.3f} {hi:>7.3f} {rb:>6.2f} "
                  f"{w:>3}/{l:<5} {raw[b]:>9.2e} {adj[b]:>9.2e}")


# -----------------------------------------------------------------------------
# Step 2: LeetCode per-problem CIs + paired bootstrap
# map grid cell -> file-name stem used by eval_lc_*.jsonl (legacy: qwenB=s10, dsA=ds)
CELL_STEM = {("QWEN", "A"): "qwenA", ("QWEN", "B"): "s10",
             ("DeepSeek", "A"): "ds", ("DeepSeek", "B"): "dsB",
             ("CodeLLama", "A"): "clA", ("CodeLLama", "B"): "clB"}
SLICE_EP = 5
BASE_KEYS = {"GA": "ga", "GD": "gd", "NPO": "npo", "DPO": "dpo",
             "SIMPNO": "simnpo", "CodeEraser": "ila", "PROD": "prod"}


def find_eval(methkey, stem):
    """Return the highest-epoch eval_lc file for a method/cell, or None."""
    cand = sorted(glob.glob(f"eval_lc_{methkey}_{stem}_ep*.jsonl"))
    if not cand:
        cand = sorted(glob.glob(f"eval_lc_{methkey}_{stem}.jsonl"))
    return cand[-1] if cand else None


def load_pass(path):
    """task_id -> {split: passed(bool)} for a eval jsonl."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["task_id"]] = (r["split"], bool(r["passed"]))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - half) / d, (c + half) / d)


def step2():
    print("\n" + "=" * 78)
    print("STEP 2  LeetCode per-problem uncertainty (pass@1)")
    print("=" * 78)
    print("\nSLICE pass@1 with exact Wilson 95% CI (per cell):")
    print(f"{'cell':12} {'split':7} {'pass@1':>7} {'95% CI':>16} {'n':>4}")
    slice_pass = {}
    for (model, split), stem in CELL_STEM.items():
        f = find_eval("cil", stem)
        if not f:
            continue
        rows = [json.loads(l) for l in open(f)]
        slice_pass[(model, split)] = {r["task_id"]: (r["split"], bool(r["passed"])) for r in rows}
        for sp in ("forget", "retain"):
            items = [r for r in rows if r["split"] == sp]
            k = sum(bool(r["passed"]) for r in items)
            n = len(items)
            lo, hi = wilson(k, n)
            print(f"{model+'-'+split:12} {sp:7} {k/max(n,1):>7.3f} "
                  f"[{lo:.3f}, {hi:.3f}]   {n:>4}")

    print("\nPaired per-problem bootstrap: retain pass@1, SLICE - baseline (95% CI)")
    print("positive = SLICE retains more; matched by task_id; 10k resamples")
    rng = np.random.default_rng(0)
    print(f"{'cell':12} {'baseline':11} {'Δ retain':>9} {'95% CI':>18} {'n':>5}")
    for (model, split), stem in CELL_STEM.items():
        sp = slice_pass.get((model, split))
        if not sp:
            continue
        s_ret = {t: p for t, (spl, p) in sp.items() if spl == "retain"}
        for b in BASELINES:
            if b == "PROD" and stem not in ("clA", "clB"):
                continue  # PROD LeetCode only for Code Llama
            bf = find_eval(BASE_KEYS[b], stem)
            if not bf:
                continue
            bp = load_pass(bf)
            b_ret = {t: p for t, (spl, p) in bp.items() if spl == "retain"}
            common = sorted(set(s_ret) & set(b_ret))
            if len(common) < 20:
                continue
            diff = np.array([int(s_ret[t]) - int(b_ret[t]) for t in common], float)
            boot = np.array([rng.choice(diff, len(diff), replace=True).mean()
                             for _ in range(10000)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"{model+'-'+split:12} {b:11} {diff.mean():>9.3f} "
                  f"[{lo:>6.3f}, {hi:>6.3f}]   {len(common):>5}")


if __name__ == "__main__":
    step1()
    step2()
