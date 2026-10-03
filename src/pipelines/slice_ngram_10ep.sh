#!/bin/bash
# Extend n-gram SLICE to 10 epochs (epochs 1-5 reproduce; eval new 6-10).
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
SPLITS=leetcode_splits_s10.json

echo "=== TRAIN slice (n-gram) 10 epochs ==="
LC_MUTANTS=lc_mutants_ngram.json \
python unlearn_lc.py --method slice --epochs 10 --splits $SPLITS \
  --outdir adapters_lc_ngram --mkey _ngram10 || exit 1

for ep in 6 7 8 9 10; do
  echo "=== EVAL slice_ngram epoch $ep ==="
  python eval_lc.py --tag slice_ngram_ep${ep} --batch-size 12 \
    --adapter adapters_lc_ngram/slice/epoch$ep --splits $SPLITS \
    || echo "EVAL slice_ngram_ep${ep} FAILED"
done

echo "=== DONE — n-gram SLICE 10-epoch trajectory (forget/heldout/retain) ==="
python - <<'PY'
import json
s=json.load(open('lc_summary.json'))
print("LLM-annotation SLICE (s10, 5 ep):")
for ep in range(1,6):
    k=f'slice_s10_ep{ep}'
    if k in s: print(f"  ep{ep}: {s[k]['forget']['pass@1']*100:.1f}/{s[k]['heldout']['pass@1']*100:.1f}/{s[k]['retain']['pass@1']*100:.1f}")
print("n-gram-annotation SLICE (s10, 10 ep):")
for ep in range(1,11):
    k=f'slice_ngram_ep{ep}'
    if k in s: print(f"  ep{ep}: {s[k]['forget']['pass@1']*100:.1f}/{s[k]['heldout']['pass@1']*100:.1f}/{s[k]['retain']['pass@1']*100:.1f}")
PY
