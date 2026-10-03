#!/bin/bash
# SLICE using n-gram-detected important lines (LLM-annotator-free), s10 split.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
SPLITS=leetcode_splits_s10.json

echo "=== GEN MUTANTS (n-gram important lines) ==="
LC_SPLITS=$SPLITS \
LC_FORGET_ANN=leetcode_forget_ngram_il.jsonl \
LC_CANON=leetcode_canon_ngram_il.jsonl \
LC_MUTANTS_OUT=lc_mutants_ngram.json \
python gen_mutants_lc.py || exit 1

echo "=== TRAIN slice (n-gram) ==="
LC_MUTANTS=lc_mutants_ngram.json \
python unlearn_lc.py --method slice --splits $SPLITS \
  --outdir adapters_lc_ngram --mkey _ngram || exit 1

for ep in 1 2 3 4 5; do
  echo "=== EVAL slice_ngram epoch $ep ==="
  python eval_lc.py --tag slice_ngram_ep${ep} --batch-size 12 \
    --adapter adapters_lc_ngram/slice/epoch$ep --splits $SPLITS \
    || echo "EVAL slice_ngram_ep${ep} FAILED"
done

echo "=== DONE — n-gram SLICE summary ==="
python - <<'PY'
import json
s=json.load(open('lc_summary.json'))
print("LLM-annotation SLICE (s10):")
for ep in range(1,6):
    k=f'slice_s10_ep{ep}'
    if k in s: print(f"  ep{ep}: {s[k]['forget']['pass@1']*100:.1f}/{s[k]['heldout']['pass@1']*100:.1f}/{s[k]['retain']['pass@1']*100:.1f}")
print("n-gram-annotation SLICE (s10):")
for ep in range(1,6):
    k=f'slice_ngram_ep{ep}'
    if k in s: print(f"  ep{ep}: {s[k]['forget']['pass@1']*100:.1f}/{s[k]['heldout']['pass@1']*100:.1f}/{s[k]['retain']['pass@1']*100:.1f}")
PY
