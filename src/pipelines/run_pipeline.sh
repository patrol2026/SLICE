#!/bin/bash
# Full unlearning pipeline: baseline eval, then train + eval each method.
cd /path/to/slice

echo "=== BASELINE EVAL (before unlearning) ==="
python eval_hf.py --tag baseline || exit 1

for m in ga gd dpo npo; do
  echo "=== TRAIN $m ==="
  python unlearn.py --method $m || { echo "TRAIN $m FAILED"; continue; }
  echo "=== EVAL $m ==="
  python eval_hf.py --tag $m --adapter adapters/$m || echo "EVAL $m FAILED"
done

echo "=== FINAL SUMMARY ==="
cat summary.json
