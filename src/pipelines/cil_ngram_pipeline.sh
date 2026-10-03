#!/bin/bash
# CIL-DPO using n-gram-detected important lines (LLM-annotator-free), s10 split.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
SPLITS=leetcode_splits_s10.json

echo "=== GEN MUTANTS (n-gram important lines) ==="
LC_SPLITS=$SPLITS \
LC_FORGET_ANN=leetcode_forget_ngram_il.jsonl \
LC_CANON=leetcode_canon_ngram_il.jsonl \
LC_MUTANTS_OUT=lc_mutants_ngram.json \
python gen_mutants_lc.py || exit 1

echo "=== TRAIN cil (n-gram) ==="
LC_MUTANTS=lc_mutants_ngram.json \
python unlearn_lc.py --method cil --splits $SPLITS \
  --outdir adapters_lc_ngram --mkey _ngram || exit 1

for ep in 1 2 3 4 5; do
  echo "=== EVAL cil_ngram epoch $ep ==="
  python eval_lc.py --tag cil_ngram_ep${ep} --batch-size 12 \
    --adapter adapters_lc_ngram/cil/epoch$ep --splits $SPLITS \
    || echo "EVAL cil_ngram_ep${ep} FAILED"
done

echo "=== DONE — n-gram CIL summary ==="
python - <<'PY'
import json
s=json.load(open('lc_summary.json'))
print("LLM-annotation CIL (s10):")
for ep in range(1,6):
    k=f'cil_s10_ep{ep}'
    if k in s: print(f"  ep{ep}: {s[k]['forget']['pass@1']*100:.1f}/{s[k]['heldout']['pass@1']*100:.1f}/{s[k]['retain']['pass@1']*100:.1f}")
print("n-gram-annotation CIL (s10):")
for ep in range(1,6):
    k=f'cil_ngram_ep{ep}'
    if k in s: print(f"  ep{ep}: {s[k]['forget']['pass@1']*100:.1f}/{s[k]['heldout']['pass@1']*100:.1f}/{s[k]['retain']['pass@1']*100:.1f}")
PY
