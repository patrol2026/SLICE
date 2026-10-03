"""Score the human-validation study.

Stage 1 (after A and B fill their sheets):
    python human_val_score.py agree
  -> line-level Cohen's kappa A vs B, per-problem disagreement report in
     human_val/disagreements.txt, and pre-filled consensus sheets in
     human_val/C (agreed lines filled; disputed lines marked '?').

Stage 2 (after adjudicating: edit human_val/C sheets, replace '?' answers):
    python human_val_score.py final
  -> kappa/F1 of the LLM annotation vs the adjudicated human consensus.
"""

import json
import re
import sys
from pathlib import Path

SRC = "leetcode_forget_generated_important_lines.jsonl"
BASE = Path("human_val")


def parse_answer(path, n_lines):
    if not path.exists():
        return None
    txt = path.read_text()
    m = re.search(r"^ANSWER:\s*(.*)$", txt, re.M)
    if not m or not m.group(1).strip():
        return None            # unanswered
    if m.group(1).strip().lower() == "none":
        return set()           # deliberately empty: no important lines
    picked = set()
    for part in m.group(1).replace(" ", "").split(","):
        if not part or part == "?":
            continue
        if "-" in part:
            a, b = part.split("-")
            picked.update(range(int(a), int(b) + 1))
        else:
            picked.add(int(part))
    return {p for p in picked if 1 <= p <= n_lines}


def content_lines(sol):
    """1-indexed non-blank, non-comment lines — the units of agreement."""
    out = []
    for i, ln in enumerate(sol.split("\n"), 1):
        s = ln.strip()
        if s and not s.startswith("#"):
            out.append(i)
    return out


def kappa(y1, y2):
    n = len(y1)
    po = sum(a == b for a, b in zip(y1, y2)) / n
    p1a, p1b = sum(y1) / n, sum(y2) / n
    pe = p1a * p1b + (1 - p1a) * (1 - p1b)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def load(stage_dirs):
    man = json.load(open(BASE / "manifest.json"))
    recs = {r["task_id"]: r for r in map(json.loads, open(SRC))}
    rows = []
    for p in man["problems"]:
        fname = f"{p['idx']:03d}_{p['task_id']}.txt"
        sol = recs[p["task_id"]]["generated_solution"]
        cl = content_lines(sol)
        answers = {}
        for d in stage_dirs:
            a = parse_answer(BASE / d / fname, p["n_lines"])
            if a is None:
                print(f"  !! {d}/{fname}: ANSWER missing — skipped")
                break
            answers[d] = a
        else:
            rows.append((p, sol, cl, answers))
    return rows, recs


def llm_lines(rec):
    s = set()
    for b in rec["important_lines"]:
        s.update(range(b["start_line"], b["end_line"] + 1))
    return s


def stage_agree():
    rows, _ = load(["A", "B"])
    ya, yb = [], []
    (BASE / "C").mkdir(exist_ok=True)
    rep = []
    for p, sol, cl, ans in rows:
        A, B = ans["A"], ans["B"]
        ya += [int(i in A) for i in cl]
        yb += [int(i in B) for i in cl]
        agreed = (A & B) | (set(cl) - A - B)
        disputed = sorted((A ^ B) & set(cl))
        if disputed:
            lines = sol.split("\n")
            rep.append(f"--- {p['idx']:03d} {p['task_id']} ---")
            for d in disputed:
                who = "A only" if d in A else "B only"
                rep.append(f"  L{d} [{who}]: {lines[d-1].strip()[:70]}")
        fname = f"{p['idx']:03d}_{p['task_id']}.txt"
        cons = sorted(A & B)
        marks = ",".join(map(str, cons)) if cons else ""
        q = ("," if cons and disputed else "") + ",".join(f"?{d}" for d in disputed)
        sheet = (BASE / "A" / fname).read_text()
        sheet = re.sub(r"^ANSWER:.*$", f"ANSWER: {marks}{q}", sheet, flags=re.M)
        (BASE / "C" / fname).write_text(sheet)
    k = kappa(ya, yb)
    n_dis = sum(a != b for a, b in zip(ya, yb))
    print(f"problems scored: {len(rows)}")
    print(f"line decisions:  {len(ya)}  (disagreements: {n_dis})")
    print(f"Cohen's kappa (A vs B): {k:.3f}")
    (BASE / "disagreements.txt").write_text("\n".join(rep) or "none")
    print("adjudication sheets in human_val/C — resolve every '?N' entry, "
          "then run: python human_val_score.py final")


def stage_final():
    rows, recs = load(["C"])
    yh, yl = [], []
    tp = fp = fn = 0
    for p, sol, cl, ans in rows:
        H = ans["C"]
        L = llm_lines(recs[p["task_id"]]) & set(cl)
        yh += [int(i in H) for i in cl]
        yl += [int(i in L) for i in cl]
        tp += len(H & L); fp += len(L - H); fn += len(H - L)
    prec = tp / (tp + fp) if tp + fp else 0
    rec_ = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0
    print(f"problems scored: {len(rows)}   line decisions: {len(yh)}")
    print(f"LLM vs human consensus:  kappa {kappa(yh, yl):.3f}   "
          f"P {prec:.3f}  R {rec_:.3f}  F1 {f1:.3f}")


if __name__ == "__main__":
    {"agree": stage_agree, "final": stage_final}[sys.argv[1]]()
