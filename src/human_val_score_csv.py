"""Score CSV-based human validation annotations.

  python human_val_score_csv.py agree
      Cohen's kappa A vs B (with 95% cluster-bootstrap CI over problems),
      writes human_val/disagreements.txt and a consensus template
      human_val/annotation_C.csv (agreed labels filled, disputes '?').

  python human_val_score_csv.py final <name> [<name2> ...]
      Scores human_val/annotation_<name>.csv (e.g. claude, gpt, gemini)
      against the adjudicated consensus annotation_C.csv: kappa, P/R/F1 with
      cluster-bootstrap CIs; with 2+ names also reports paired F1-difference
      bootstrap between consecutive pairs.
"""

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path("human_val")
B = 2000  # bootstrap resamples


def read(name):
    """-> {(problem_no, line_no): 0/1} for content rows; error on blanks."""
    out, missing = {}, 0
    with open(BASE / f"annotation_{name}.csv") as f:
        for row in csv.DictReader(f):
            if row["important"].strip().lower() == "skip":
                continue
            v = row["important"].strip()
            key = (int(row["problem_no"]), int(row["line_no"]))
            if v in ("0", "1"):
                out[key] = int(v)
            elif v == "?":
                out[key] = "?"
            else:
                missing += 1
    if missing:
        print(f"  !! annotation_{name}.csv: {missing} content rows unlabeled")
    return out


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    p1a = sum(a for a, _ in pairs) / n
    p1b = sum(b for _, b in pairs) / n
    pe = p1a * p1b + (1 - p1a) * (1 - p1b)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def by_problem(x, y):
    g = defaultdict(list)
    for k in x:
        if k in y and x[k] != "?" and y[k] != "?":
            g[k[0]].append((x[k], y[k]))
    return g


def cboot(groups, stat):
    probs = list(groups)
    rng = random.Random(0)
    vals = []
    for _ in range(B):
        sel = [groups[rng.choice(probs)] for _ in probs]
        vals.append(stat([p for g in sel for p in g]))
    vals.sort()
    return vals[int(0.025 * B)], vals[int(0.975 * B)]


def prf(pairs):  # pairs = (gold, pred)
    tp = sum(1 for g, p in pairs if g and p)
    fp = sum(1 for g, p in pairs if not g and p)
    fn = sum(1 for g, p in pairs if g and not p)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return 2 * P * R / (P + R) if P + R else 0.0, P, R


def stage_agree():
    A, Bb = read("A"), read("B")
    g = by_problem(A, Bb)
    allp = [p for grp in g.values() for p in grp]
    k = kappa(allp)
    lo, hi = cboot(g, kappa)
    print(f"problems: {len(g)}   line decisions: {len(allp)}")
    print(f"Cohen's kappa (A vs B): {k:.3f}   95% cluster-bootstrap CI "
          f"[{lo:.3f}, {hi:.3f}]")
    # consensus template + disagreement report
    rows = list(csv.DictReader(open(BASE / "annotation_A.csv")))
    rep = []
    for row in rows:
        if row["important"].strip().lower() == "skip":
            continue
        key = (int(row["problem_no"]), int(row["line_no"]))
        a, b = A.get(key), Bb.get(key)
        if a == b:
            row["important"] = str(a)
        else:
            row["important"] = "?"
            rep.append(f"p{key[0]:03d} L{key[1]:3d} A={a} B={b}: "
                       f"{row['code'].strip()[:70]}")
    with open(BASE / "annotation_C.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    (BASE / "disagreements.txt").write_text("\n".join(rep) or "none")
    print(f"disagreements: {len(rep)} -> human_val/disagreements.txt")
    print("adjudicate: edit annotation_C.csv, replace every '?' with 0/1, "
          "then run: python human_val_score_csv.py final <agent>")


def stage_final(names):
    gold = read("C")
    if any(v == "?" for v in gold.values()):
        sys.exit("annotation_C.csv still contains '?' — finish adjudication")
    per = {}
    for nm in names:
        pred = read(nm)
        g = by_problem(gold, pred)
        allp = [p for grp in g.values() for p in grp]
        f1, P, R = prf(allp)
        klo, khi = cboot(g, kappa)
        flo, fhi = cboot(g, lambda ps: prf(ps)[0])
        print(f"{nm:>8}: kappa {kappa(allp):.3f} [{klo:.3f},{khi:.3f}]   "
              f"P {P:.3f}  R {R:.3f}  F1 {f1:.3f} [{flo:.3f},{fhi:.3f}]  "
              f"({len(allp)} decisions)")
        per[nm] = g
    # paired comparison between consecutive names
    for x, y in zip(names, names[1:]):
        probs = sorted(set(per[x]) & set(per[y]))
        rng = random.Random(0)
        diffs = []
        for _ in range(B):
            sel = [rng.choice(probs) for _ in probs]
            dx = prf([p for s in sel for p in per[x][s]])[0]
            dy = prf([p for s in sel for p in per[y][s]])[0]
            diffs.append(dx - dy)
        diffs.sort()
        lo, hi = diffs[int(0.025 * B)], diffs[int(0.975 * B)]
        sig = "significant" if lo > 0 or hi < 0 else "not significant"
        print(f"paired ΔF1 {x}-{y}: [{lo:+.3f}, {hi:+.3f}]  ({sig})")


if __name__ == "__main__":
    if sys.argv[1] == "agree":
        stage_agree()
    else:
        stage_final(sys.argv[2:])
