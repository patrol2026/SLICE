#!/bin/bash
# Ablation study of the SLICE loss (s10 split, LLM-annotation mutants).
#   noil     : preference over whole solution (no important-line masking)
#   noforget : drop forget-preference term (guard + retain NLL only)
#   noguard  : drop guard pairs (forget-pref + retain NLL only)
#   nonll    : drop retain NLL anchor (forget-pref + guard only)
# Compares against the full SLICE (slice_s10).
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
SPLITS=leetcode_splits_s10.json

# wait for the running 10-epoch n-gram job to release the GPU
while pgrep -f slice_ngram_10ep.sh >/dev/null || pgrep -f "eval_lc.py --tag slice_ngram" >/dev/null; do
  sleep 60
done
echo "=== GPU free — starting ablations ==="

for AB in noil noforget noguard nonll; do
  echo "=== TRAIN slice ablation=$AB ==="
  LC_MUTANTS=lc_mutants_ngram.json \
  python unlearn_lc.py --method slice --ablation $AB --epochs 5 --splits $SPLITS \
    --outdir adapters_ab_$AB --mkey _ab_$AB \
    || { echo "TRAIN $AB FAILED"; continue; }
  for ep in 1 3 5; do
    echo "=== EVAL slice_$AB epoch $ep ==="
    python eval_lc.py --tag slice_${AB}_ep${ep} --batch-size 12 \
      --adapter adapters_ab_$AB/slice/epoch$ep --splits $SPLITS \
      || echo "EVAL slice_${AB}_ep${ep} FAILED"
  done
done

echo "=== ABLATION SUMMARY (forget/heldout/retain) ==="
python - <<'PY'
import json
s=json.load(open('lc_summary.json'))
def row(tag):
    v=s.get(tag)
    return f"{v['forget']['pass@1']*100:.1f}/{v['heldout']['pass@1']*100:.1f}/{v['retain']['pass@1']*100:.1f}" if v else "--"
print("full SLICE n-gram (slice_ngram): ep1", row('slice_ngram_ep1'), " ep3", row('slice_ngram_ep3'), " ep5", row('slice_ngram_ep5'))
for ab in ['noil','noforget','noguard','nonll']:
    print(f"{ab:24}   ep1", row(f'slice_{ab}_ep1'), " ep3", row(f'slice_{ab}_ep3'), " ep5", row(f'slice_{ab}_ep5'))
PY
