"""Language-agnostic important-line detectors via tree-sitter (Python/Java/C++).

ast_scores      — score lines by AST node categories (operators, predicates,
                  calls, control) — SlimCode-style, works across languages.
dataflow_scores — def-use centrality: a line using/defining widely-used
                  identifiers scores high (GraphCodeBERT-style data flow).
"""

import json
import math
import re
import tree_sitter_python, tree_sitter_java, tree_sitter_cpp
from collections import Counter
from tree_sitter import Language, Parser

_LANG = {
    "python": Language(tree_sitter_python.language()),
    "java": Language(tree_sitter_java.language()),
    "cpp": Language(tree_sitter_cpp.language()),
}
_PARSER = {k: Parser(v) for k, v in _LANG.items()}

# node types that carry problem-specific logic, per language
_IMPORTANT = {
    "python": {"binary_operator": 2.0, "comparison_operator": 2.0,
               "boolean_operator": 1.5, "call": 0.8, "subscript": 1.0,
               "conditional_expression": 1.5, "augmented_assignment": 1.0,
               "return_statement": 0.6, "list_comprehension": 1.5},
    "java": {"binary_expression": 2.0, "method_invocation": 0.8,
             "array_access": 1.0, "ternary_expression": 1.5,
             "update_expression": 1.0, "return_statement": 0.6,
             "unary_expression": 0.8},
    "cpp": {"binary_expression": 2.0, "call_expression": 0.8,
            "subscript_expression": 1.0, "conditional_expression": 1.5,
            "update_expression": 1.0, "return_statement": 0.6,
            "unary_expression": 0.8},
}
_SCAFFOLD = {"import_statement", "import_from_statement", "package_declaration",
             "import_declaration", "preproc_include", "using_declaration",
             "class_definition", "function_definition", "method_declaration",
             "class_declaration"}


def _parse(code, lang):
    return _PARSER[lang].parse(bytes(code, "utf8")).root_node


# ---- n-gram / TF-IDF rarity detector (model-free, corpus-based) ------------
_SCAFFOLD_RE = re.compile(
    r"^\s*(import |from |package |using |#include|#|@|//|/\*|\*|"
    r">>>|\"\"\"|'''|public |private |class |def |void |int |"
    r"return$|\{|\}|$)")
_IDF_CACHE = {}


def _idf_for(lang, n=3):
    """Char n-gram document frequency over all solved solutions in this lang."""
    if lang not in _IDF_CACHE:
        df, N = Counter(), 0
        for l in open(f"results_hex_{lang}.jsonl"):
            r = json.loads(l)
            if not r.get("passed"):
                continue
            for line in r["generated_code"].split("\n"):
                s = re.sub(r"\s+", " ", line.strip())
                if not s:
                    continue
                for g in set(s[i:i + n] for i in range(max(0, len(s) - n + 1))):
                    df[g] += 1
                N += 1
        _IDF_CACHE[lang] = (df, max(N, 1))
    return _IDF_CACHE[lang]


def tfidf_scores(code, lang, n=3):
    """Line rarity: mean inverse-document-frequency of a line's char n-grams,
    measured against a per-language corpus of solved solutions. Model-free."""
    df, N = _idf_for(lang, n)
    lines = code.split("\n")
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


def ast_scores(code, lang):
    lines = code.split("\n")
    scores = {i + 1: 0.0 for i in range(len(lines))}
    weights = _IMPORTANT[lang]
    root = _parse(code, lang)
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in weights:
            ln = n.start_point[0] + 1
            if ln in scores:
                scores[ln] += weights[n.type]
        stack.extend(n.children)
    return scores


def dataflow_scores(code, lang):
    """Def-use centrality: identifiers appearing on many lines are central;
    a line's score = sum over its identifiers of (#lines that identifier spans)."""
    lines = code.split("\n")
    scores = {i + 1: 0.0 for i in range(len(lines))}
    root = _parse(code, lang)
    id_lines = {}      # identifier -> set of lines
    per_line_ids = {}  # line -> list of identifiers
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "identifier":
            name = n.text.decode("utf8", "ignore")
            ln = n.start_point[0] + 1
            id_lines.setdefault(name, set()).add(ln)
            per_line_ids.setdefault(ln, []).append(name)
        stack.extend(n.children)
    for ln, ids in per_line_ids.items():
        if ln in scores:
            # centrality = how many other lines share this line's variables
            scores[ln] = sum(len(id_lines[i]) - 1 for i in ids)
    return scores


if __name__ == "__main__":
    import json
    samples = {"python": None, "java": None, "cpp": None}
    for lang in samples:
        for l in open(f"results_hex_{lang}.jsonl"):
            r = json.loads(l)
            if r["passed"]:
                samples[lang] = r; break
    for lang, r in samples.items():
        code = r["generated_code"]
        a = ast_scores(code, lang); d = dataflow_scores(code, lang)
        na = sorted(a, key=lambda k: -a[k])[:3]
        nd = sorted(d, key=lambda k: -d[k])[:3]
        print(f"\n=== {lang} {r['task_id']} ===")
        lines = code.split("\n")
        print("AST top:", [(k, round(a[k], 1)) for k in na])
        print("DF  top:", [(k, round(d[k], 1)) for k in nd])
        for k in sorted(set(na) | set(nd)):
            print(f"  L{k}: {lines[k-1].strip()[:70]}")
