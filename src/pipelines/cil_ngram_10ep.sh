#!/bin/bash
# Extend n-gram CIL to 10 epochs (epochs 1-5 reproduce; eval new 6-10).
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
SPLITS=leetcode_splits_s10.json

echo "=== TRAIN cil (n-gram) 10 epochs ==="
LC_MUTANTS=lc_mutants_ngram.json \
python unlearn_lc.py --method cil --epochs 10 --splits $SPLITS \
  --outdir adapters_lc_ngram --mkey _ngram10 || exit 1

for ep in 6 7 8 9 10; do
  echo "=== EVAL cil_ngram epoch $ep ==="
  python eval_lc.py --tag cil_ngram_ep${ep} --batch-size 12 \
    --adapter adapters_lc_ngram/cil/epoch$ep --splits $SPLITS \
    || echo "EVAL cil_ngram_ep${ep} FAILED"
done

echo "=== DONE — n-gram CIL 10-epoch trajectory (forget/heldout/retain) ==="
python - <<'PY'
import json
s=json.load(open('lc_summary.json'))
print("LLM-annotation CIL (s10, 5 ep):")
for ep in range(1,6):
    k=f'cil_s10_ep{ep}'
    if k in s: print(f"  ep{ep}: {s[k]['forget']['pass@1']*100:.1f}/{s[k]['heldout']['pass@1']*100:.1f}/{s[k]['retain']['pass@1']*100:.1f}")
print("n-gram-annotation CIL (s10, 10 ep):")
for ep in range(1,11):
    k=f'cil_ngram_ep{ep}'
    if k in s: print(f"  ep{ep}: {s[k]['forget']['pass@1']*100:.1f}/{s[k]['heldout']['pass@1']*100:.1f}/{s[k]['retain']['pass@1']*100:.1f}")
PY
