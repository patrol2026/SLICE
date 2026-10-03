#!/bin/bash
# Full LeetCode unlearning pipeline: baseline eval, then train+eval all methods.
# cil runs last so gen_mutants_lc.py (running in parallel) has time to finish.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== BASELINE EVAL ==="
python eval_lc.py --tag baseline || exit 1

for m in ga gd dpo npo ila cil; do
  if [ "$m" = "cil" ] && [ ! -f lc_mutants.json ]; then
    echo "cil SKIPPED: lc_mutants.json missing"
    continue
  fi
  echo "=== TRAIN $m ==="
  python unlearn_lc.py --method $m || { echo "TRAIN $m FAILED"; continue; }
  echo "=== EVAL $m ==="
  python eval_lc.py --tag $m --adapter adapters_lc/$m || echo "EVAL $m FAILED"
done

echo "=== FINAL SUMMARY ==="
cat lc_summary.json
echo "=== METRICS ==="
cat lc_metrics.json
