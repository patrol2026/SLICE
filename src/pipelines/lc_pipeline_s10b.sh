#!/bin/bash
# Resume the s10 run: finish gd ep5 eval, then dpo/npo/codeeraser/slice train + per-epoch
# evals. Skip rule: an epoch scoring 0/0/0 kills the remaining epochs.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
SPLITS=leetcode_splits_s10.json

all_zero() {
  python - "$1" <<'PY'
import json, sys
s = json.load(open("lc_summary.json")).get(sys.argv[1])
sys.exit(0 if s and sum(v["passed"] for v in s.values()) == 0 else 1)
PY
}

echo "=== EVAL gd_s10 epoch 5 (resumed) ==="
python eval_lc.py --tag gd_s10_ep5 --adapter adapters_lc_s10/gd/epoch5 \
  --splits $SPLITS || echo "EVAL gd_s10_ep5 FAILED"

for m in dpo npo codeeraser slice; do
  if [ ! -d adapters_lc_s10/$m ]; then
    echo "=== TRAIN $m (s10) ==="
    python unlearn_lc.py --method $m --splits $SPLITS \
      --outdir adapters_lc_s10 --mkey _s10 \
      || { echo "TRAIN $m FAILED"; continue; }
  fi
  for ep in 1 2 3 4 5; do
    echo "=== EVAL ${m}_s10 epoch $ep ==="
    python eval_lc.py --tag ${m}_s10_ep${ep} \
      --adapter adapters_lc_s10/$m/epoch$ep --splits $SPLITS \
      || { echo "EVAL ${m}_s10_ep${ep} FAILED"; continue; }
    if all_zero "${m}_s10_ep${ep}"; then
      echo "=== ${m}_s10 epoch $ep is 0/0/0 — skipping remaining epochs ==="
      break
    fi
  done
done

echo "=== FINAL SUMMARY ==="
cat lc_summary.json
