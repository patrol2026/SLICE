#!/bin/bash
# Resume the LC pipeline: finish CodeEraser eval (resumable), then SLICE train + eval.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== EVAL codeeraser (resumed) ==="
python eval_lc.py --tag codeeraser --adapter adapters_lc/codeeraser || echo "EVAL codeeraser FAILED"

echo "=== TRAIN slice ==="
python unlearn_lc.py --method slice || { echo "TRAIN slice FAILED"; exit 1; }
echo "=== EVAL slice ==="
python eval_lc.py --tag slice --adapter adapters_lc/slice || echo "EVAL slice FAILED"

echo "=== FINAL SUMMARY ==="
cat lc_summary.json
echo "=== METRICS ==="
cat lc_metrics.json
