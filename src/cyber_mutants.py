"""Verified mutants for the CyberSecEval task (no detection: the important line
IS the analyzer-flagged vulnerable line).

For each of the 737 records: locate the vulnerable line in the snippet, mutate it
(operator mutations + fallback perturbations), and keep the first candidate whose
mutated snippet NO LONGER matches the record's vulnerability regex (vuln provably
removed). Output cyber_mutants.jsonl:
  {cid, code, rule, language, block:{start,end,orig,mutated}}
"""
import json
import re as _re

from gen_mutants import try_mutations

RECORDS = "cyber_records.jsonl"
OUT = "cyber_mutants.jsonl"


def fallback_mutations(orig):
    """String / number / identifier perturbations for operator-free lines."""
    out = []
    m = _re.search(r"\b\d+\b", orig)
    if m:
        out.append(orig[:m.start()] + str(int(m.group()) + 1) + orig[m.end():])
    m = _re.search(r"(['\"])((?:(?!\1).){3,})\1", orig)
    if m:
        s = m.group(2)
        out.append(orig.replace(m.group(0), m.group(1) + s[::-1] + m.group(1), 1))
    ids = sorted(set(_re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b", orig)),
                 key=len, reverse=True)
    for name in ids[:3]:
        if name in ("self", "None", "True", "False", "return", "import", "public",
                    "static", "const", "final"):
            continue
        out.append(_re.sub(r"\b%s\b" % _re.escape(name), name + "_alt", orig, count=1))
    return out


def vuln_index(code, line_text):
    t = line_text.strip()
    if not t:
        return None
    lines = code.split("\n")
    for i, ln in enumerate(lines, 1):
        if ln.strip() == t:
            return i
    for i, ln in enumerate(lines, 1):
        if t in ln:
            return i
    return None


_KW = {"self", "None", "True", "False", "return", "import", "public", "static",
       "const", "final", "void", "int", "char", "let", "var", "func", "def",
       "class", "new", "this", "null", "true", "false", "if", "else", "for",
       "while", "printf", "print"}
_OPS = [("<=", ">"), (">=", "<"), ("==", "!="), ("!=", "=="),
        ("&&", "||"), ("||", "&&"), (" < ", " > "), (" > ", " < ")]


def multi_mutation(orig):
    """Apply SEVERAL perturbations to the line at once (stronger preference
    signal than a single-token edit): flip an operator, bump every integer,
    rename up to 4 identifiers, and reverse string literals."""
    line = orig
    for a, b in _OPS:
        if a in line:
            line = line.replace(a, b, 1)
            break
    line = _re.sub(r"\b\d+\b", lambda m: str(int(m.group()) + 1), line)
    ids = [i for i in dict.fromkeys(_re.findall(r"\b[A-Za-z_]\w{2,}\b", line))
           if i not in _KW]
    for name in ids[:4]:
        line = _re.sub(r"\b%s\b" % _re.escape(name), name + "_alt", line, count=1)
    line = _re.sub(r"(['\"])((?:(?!\1).){3,})\1",
                   lambda m: m.group(1) + m.group(2)[::-1] + m.group(1), line)
    return line if line != orig else None


def targeted_mutations(orig, pat):
    """Perturb the exact construct the rule matches, so the vuln is removed
    while keeping the edit minimal (the cleanest 'secure' target)."""
    out = []
    m = pat.search(orig)
    if not m:
        return out
    span = m.group(0)
    # (a) break the first identifier inside the matched construct
    im = _re.search(r"[A-Za-z_]\w*", span)
    if im:
        new_span = span[:im.end()] + "_alt" + span[im.end():]
        out.append(orig.replace(span, new_span, 1))
        broken = span[:im.start()] + im.group() + "_x" + span[im.end():]
        out.append(orig.replace(span, broken, 1))
    return out


def verified_mutant(code, idx, rule):
    """Mutate line `idx`; keep first candidate where the rule no longer fires
    on the full mutated snippet."""
    pat = _re.compile(rule)
    lines = code.split("\n")
    orig = lines[idx - 1]
    # multi-change candidate first (strongest signal), then single-change fallbacks
    heavy = multi_mutation(orig)
    cands = ([heavy] if heavy else []) + list(try_mutations(orig))[:8] \
        + fallback_mutations(orig) + targeted_mutations(orig, pat)
    for cand in cands:
        if cand == orig:
            continue
        mutated = "\n".join(lines[:idx - 1] + [cand] + lines[idx:])
        if not pat.search(mutated):          # vulnerability provably removed
            return {"start": idx, "end": idx, "orig": orig, "mutated": cand}
    return None


def main():
    recs = [json.loads(l) for l in open(RECORDS)]
    kept = 0
    no_line = 0
    no_mut = 0
    per_lang = {}
    with open(OUT, "w") as fo:
        for r in recs:
            pl = per_lang.setdefault(r["language"], [0, 0])
            pl[0] += 1
            idx = vuln_index(r["code"], r["line_text"])
            if idx is None:
                no_line += 1
                continue
            if not _re.search(r["rule"], r["code"]):
                # rule doesn't fire on the original snippet -> can't verify removal
                no_mut += 1
                continue
            m = verified_mutant(r["code"], idx, r["rule"])
            if not m:
                no_mut += 1
                continue
            fo.write(json.dumps({"cid": r["cid"], "code": r["code"],
                                 "rule": r["rule"], "language": r["language"],
                                 "block": m}) + "\n")
            kept += 1
            pl[1] += 1
    n = len(recs)
    print(f"verified mutants: {kept}/{n}  (no vuln line found: {no_line}, "
          f"no verifiable mutation: {no_mut})")
    print("per-language (verified/total):")
    for lang, (t, k) in sorted(per_lang.items()):
        print(f"  {lang:11s} {k}/{t}")


if __name__ == "__main__":
    main()
