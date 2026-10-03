#!/bin/bash
# Train SLICE (corrective important-line DPO) + evaluate every epoch.
cd /path/to/slice
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== TRAIN slice ==="
python unlearn.py --method slice --batch-size 1 --grad-accum 8 || exit 1
echo "=== EVAL slice (final) ==="
python eval_hf.py --tag slice --adapter adapters/slice || echo "EVAL slice FAILED"
for ep in 1 2 3 4; do
  echo "=== EVAL slice epoch $ep ==="
  python eval_hf.py --tag slice_ep${ep} --adapter adapters/slice/epoch${ep} \
    || echo "EVAL slice_ep${ep} FAILED"
done

echo "=== FINAL SUMMARY ==="
cat summary.json
