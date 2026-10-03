#!/bin/bash
# Resume the LC pipeline: finish CodeEraser eval (resumable), then CIL train + eval.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== EVAL ila (resumed) ==="
python eval_lc.py --tag ila --adapter adapters_lc/ila || echo "EVAL ila FAILED"

echo "=== TRAIN cil ==="
python unlearn_lc.py --method cil || { echo "TRAIN cil FAILED"; exit 1; }
echo "=== EVAL cil ==="
python eval_lc.py --tag cil --adapter adapters_lc/cil || echo "EVAL cil FAILED"

echo "=== FINAL SUMMARY ==="
cat lc_summary.json
echo "=== METRICS ==="
cat lc_metrics.json
