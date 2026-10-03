#!/bin/bash
# 1) Train the new ila method (important-line ascent) + final eval.
# 2) Retrain ga/gd/npo with per-epoch checkpointing (dpo already has it).
# 3) Evaluate every epoch-1..4 checkpoint of every method.
# Epoch-5 evals are skipped: adapters/<m> (final) is already evaluated as eval_<m>.
cd /path/to/slice
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== TRAIN ila ==="
python unlearn.py --method ila --batch-size 1 --grad-accum 8 \
  || echo "TRAIN ila FAILED"
echo "=== EVAL ila (final) ==="
python eval_hf.py --tag ila --adapter adapters/ila || echo "EVAL ila FAILED"

for m in ga gd npo; do
  if [ ! -d adapters/$m/epoch1 ]; then
    echo "=== RETRAIN $m (per-epoch checkpoints) ==="
    python unlearn.py --method $m --batch-size 1 --grad-accum 8 \
      || echo "TRAIN $m FAILED"
  fi
done

for m in ila ga gd dpo npo; do
  for ep in 1 2 3 4; do
    if [ -d adapters/$m/epoch$ep ]; then
      echo "=== EVAL ${m} epoch $ep ==="
      python eval_hf.py --tag ${m}_ep${ep} --adapter adapters/$m/epoch$ep \
        || echo "EVAL ${m}_ep${ep} FAILED"
    fi
  done
done

echo "=== FINAL SUMMARY ==="
cat summary.json
