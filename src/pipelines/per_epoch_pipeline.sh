#!/bin/bash
# 1) Train the new codeeraser method (important-line ascent) + final eval.
# 2) Retrain ga/gd/npo with per-epoch checkpointing (dpo already has it).
# 3) Evaluate every epoch-1..4 checkpoint of every method.
# Epoch-5 evals are skipped: adapters/<m> (final) is already evaluated as eval_<m>.
cd /path/to/slice
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== TRAIN codeeraser ==="
python unlearn.py --method codeeraser --batch-size 1 --grad-accum 8 \
  || echo "TRAIN codeeraser FAILED"
echo "=== EVAL codeeraser (final) ==="
python eval_hf.py --tag codeeraser --adapter adapters/codeeraser || echo "EVAL codeeraser FAILED"

for m in ga gd npo; do
  if [ ! -d adapters/$m/epoch1 ]; then
    echo "=== RETRAIN $m (per-epoch checkpoints) ==="
    python unlearn.py --method $m --batch-size 1 --grad-accum 8 \
      || echo "TRAIN $m FAILED"
  fi
done

for m in codeeraser ga gd dpo npo; do
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
