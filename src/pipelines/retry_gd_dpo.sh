#!/bin/bash
# Retrain the two methods that OOM'd, with memory fixes + batch size 1.
cd /path/to/slice
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for m in gd dpo; do
  echo "=== TRAIN $m ==="
  python unlearn.py --method $m --batch-size 1 --grad-accum 8 \
    || { echo "TRAIN $m FAILED"; continue; }
  echo "=== EVAL $m ==="
  python eval_hf.py --tag $m --adapter adapters/$m || echo "EVAL $m FAILED"
done

echo "=== FINAL SUMMARY ==="
cat summary.json
