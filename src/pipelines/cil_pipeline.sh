#!/bin/bash
# Train CIL-DPO (corrective important-line DPO) + evaluate every epoch.
cd /path/to/slice
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== TRAIN cil ==="
python unlearn.py --method cil --batch-size 1 --grad-accum 8 || exit 1
echo "=== EVAL cil (final) ==="
python eval_hf.py --tag cil --adapter adapters/cil || echo "EVAL cil FAILED"
for ep in 1 2 3 4; do
  echo "=== EVAL cil epoch $ep ==="
  python eval_hf.py --tag cil_ep${ep} --adapter adapters/cil/epoch${ep} \
    || echo "EVAL cil_ep${ep} FAILED"
done

echo "=== FINAL SUMMARY ==="
cat summary.json
