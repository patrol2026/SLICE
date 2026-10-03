"""Important-line detectors for code solutions.

Each detector maps a solution (list of source lines) to a per-line score
dict {line_no (1-indexed): float}. Higher = more important. Binarization to a
line set is done downstream at a matched budget.

Detectors:
  mutation      — behavioral: a line is important if mutating it makes tests fail
  ablation      — behavioral: importance = tests break when the line is deleted
  structural    — static AST/token categories (SlimCode-style), model+test free
  surprisal     — per-line mean token cross-entropy under a code LM
  excess        — Rho-1 style: code-LM loss minus general-LM loss per token
  ngram         — corpus n-gram rarity (TF-IDF-like), model+test free

Model-based detectors (surprisal, excess) are computed in run_detectors.py where
the models live; here we expose the pure/static/behavioral ones.
"""

import ast
import io
import re
import tokenize
from collections import Counter

from gen_mutants import try_mutations
from run_leetcode import run_tests


# ----------------------------------------------------------------------------
# Behavioral detectors (need the test suite)
# ----------------------------------------------------------------------------

def mutation_scores(rec, pool=None):
    """Per-line importance = does ANY single-line mutation flip pass->fail.

    Score = fraction of that line's attempted mutations that break the tests
    (0 if none / line not mutable). Behavioral, model-free, needs tests.
    """
    lines = rec["generated_solution"].split("\n")
    test, ep = rec["test"], rec["entry_point"]
    scores = {i + 1: 0.0 for i in range(len(lines))}

    jobs = []  # (line_no, mutated_full_source)
    for i, ln in enumerate(lines):
        cands = list(try_mutations(ln))
        for cand in cands[:4]:
            mutated = "\n".join(lines[:i] + [cand] + lines[i + 1:])
            try:
                ast.parse(mutated)
            except SyntaxError:
                continue
            jobs.append((i + 1, mutated))

    def run(job):
        ln_no, src = job
        passed, _ = run_tests(src, test, ep)
        return ln_no, (0 if passed else 1)  # tests fail -> mutation mattered

    if pool is not None:
        results = list(pool.map(run, jobs))
    else:
        results = [run(j) for j in jobs]

    tried = Counter()
    broke = Counter()
    for ln_no, killed in results:
        tried[ln_no] += 1
        broke[ln_no] += killed
    for ln_no in tried:
        scores[ln_no] = broke[ln_no] / tried[ln_no]
    return scores


def ablation_scores(rec, pool=None):
    """SIVAND-lite: importance = tests break when the line is deleted.

    Score = 1 if deleting the line (and its exact block, naively) makes the
    solution stop passing, else 0. Behavioral, model-free, needs tests.
    Note: over-flags because deletion often causes NameError/Syntax.
    """
    lines = rec["generated_solution"].split("\n")
    test, ep = rec["test"], rec["entry_point"]
    scores = {i + 1: 0.0 for i in range(len(lines))}

    jobs = []
    for i, ln in enumerate(lines):
        if not ln.strip() or ln.strip().startswith("#"):
            continue
        reduced = "\n".join(lines[:i] + lines[i + 1:])
        jobs.append((i + 1, reduced))

    def run(job):
        ln_no, src = job
        try:
            ast.parse(src)
        except SyntaxError:
            return ln_no, 1  # removal breaks syntax -> structurally needed
        passed, _ = run_tests(src, test, ep)
        return ln_no, (0 if passed else 1)

    results = list(pool.map(run, jobs)) if pool is not None else [run(j) for j in jobs]
    for ln_no, killed in results:
        scores[ln_no] = float(killed)
    return scores


# ----------------------------------------------------------------------------
# Static structural detector (SlimCode-style, AST + token categories)
# ----------------------------------------------------------------------------

# SlimCode empirical ranking (adapted to Python): signature > identifiers >
# control ~ calls > symbols. We add data-flow (def-use) centrality on top.
_CTRL = {"if", "elif", "else", "for", "while", "return", "yield", "break",
         "continue", "and", "or", "not", "in", "is"}
_SCAFFOLD_RE = re.compile(r"^\s*(import |from |class |def |#|@|$)")


def structural_scores(rec):
    """Static line importance from AST/token categories + def-use centrality.

    Model-free, test-free. Higher for lines with operators/predicates/calls that
    are depended upon; lower for imports, signatures, trivial init / bare return.
    """
    src = rec["generated_solution"]
    lines = src.split("\n")
    scores = {i + 1: 0.0 for i in range(len(lines))}

    # token-category score per line
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        toks = []
    for t in toks:
        ln = t.start[0]
        if ln not in scores:
            continue
        s = 0.0
        if t.type == tokenize.OP and t.string in {"<", ">", "<=", ">=", "==",
                                                  "!=", "+", "-", "*", "/", "%",
                                                  "//", "**", "&", "|", "^"}:
            s += 2.0                       # operators = problem logic
        elif t.type == tokenize.NAME and t.string in _CTRL:
            s += 1.0                       # control keywords
        elif t.type == tokenize.NAME:
            s += 0.4                       # identifiers
        elif t.type == tokenize.NUMBER:
            s += 0.8                       # literals often problem-specific
        scores[ln] += s

    # def-use centrality: a line that defines a variable used by many later
    # lines is structurally central.
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None
    if tree is not None:
        defs = {}   # var -> def line
        uses = Counter()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    defs.setdefault(node.id, getattr(node, "lineno", None))
                elif isinstance(node.ctx, ast.Load):
                    uses[node.id] += 1
        for var, dln in defs.items():
            if dln in scores:
                scores[dln] += 0.5 * uses.get(var, 0)

    # zero-out pure scaffolding lines
    for i, ln in enumerate(lines):
        if _SCAFFOLD_RE.match(ln):
            scores[i + 1] = 0.0
        if ln.strip() in {"return", "pass"} or re.match(r"^\s*return \w+\s*$", ln):
            scores[i + 1] *= 0.3           # bare accumulator return
    return scores


# ----------------------------------------------------------------------------
# n-gram / TF-IDF rarity detector (corpus-based, model+test free)
# ----------------------------------------------------------------------------

def build_ngram_idf(corpus_lines, n=3):
    """Document-frequency of char n-grams across a corpus of code lines."""
    df = Counter()
    N = 0
    for line in corpus_lines:
        s = re.sub(r"\s+", " ", line.strip())
        if not s:
            continue
        grams = set(s[i:i + n] for i in range(max(0, len(s) - n + 1)))
        for g in grams:
            df[g] += 1
        N += 1
    return df, max(N, 1)


def ngram_scores(rec, df, N, n=3):
    """Line rarity: mean inverse-document-frequency of its char n-grams.

    A line whose n-grams are rare across the corpus is problem-specific.
    Model-free, test-free, language-agnostic.
    """
    import math
    lines = rec["generated_solution"].split("\n")
    scores = {i + 1: 0.0 for i in range(len(lines))}
    for i, line in enumerate(lines):
        s = re.sub(r"\s+", " ", line.strip())
        if not s or _SCAFFOLD_RE.match(line):
            continue
        grams = [s[j:j + n] for j in range(max(0, len(s) - n + 1))]
        if not grams:
            continue
        idf = [math.log(N / (1 + df.get(g, 0))) for g in grams]
        scores[i + 1] = sum(idf) / len(idf)
    return scores
