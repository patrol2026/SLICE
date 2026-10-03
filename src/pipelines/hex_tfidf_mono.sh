#!/bin/bash
# TF-IDF detector, Python-only (mono) unlearning; eval on Py/Java/C++.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== MUTANTS (tfidf) ==="
python3 -u hex_mutants.py tfidf || exit 1

echo "=== TRAIN tfidf/mono (python-only) ==="
python3 -u hex_unlearn.py --detector tfidf --mode mono --epochs 5 \
  || { echo "TRAIN FAILED"; exit 1; }

echo "=== EVAL tfidf/mono ==="
python3 -u hex_eval.py --tag tfidf_mono --adapter adapters_hex/tfidf_mono \
  || echo "EVAL FAILED"

echo "=== TF-IDF mono forget pass@1 (py / java / cpp) vs AST mono ==="
python3 - <<'PY'
import json
s = json.load(open('hex_summary.json'))
def row(tag):
    v = s.get(tag, {}).get('forget', {})
    p = lambda x: f"{x*100:.0f}" if isinstance(x,(int,float)) else "--"
    return f"{p(v.get('python'))} / {p(v.get('java'))} / {p(v.get('cpp'))}"
print(f"{'baseline (no unlearn)':26} {row('baseline')}")
print(f"{'ast · python-only':26} {row('ast_mono')}")
print(f"{'tfidf · python-only':26} {row('tfidf_mono')}")
r = s.get('tfidf_mono',{}).get('retain',{})
print(f"retain (py/java/cpp): {r.get('python')}/{r.get('java')}/{r.get('cpp')}")
PY
echo "=== DONE ==="
