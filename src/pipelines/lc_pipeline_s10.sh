#!/bin/bash
# 10/20/70 split run: train all methods, evaluate EVERY epoch checkpoint.
# Economy rule: once an epoch evaluates to 0/0/0, later epochs of that method
# are provably dead and are skipped.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
SPLITS=leetcode_splits_s10.json

all_zero() {  # returns 0 (true) if the given summary tag is 0/0/0
  python - "$1" <<'PY'
import json, sys
s = json.load(open("lc_summary.json")).get(sys.argv[1])
sys.exit(0 if s and sum(v["passed"] for v in s.values()) == 0 else 1)
PY
}

for m in ga gd dpo npo ila cil; do
  echo "=== TRAIN $m (s10) ==="
  python unlearn_lc.py --method $m --splits $SPLITS \
    --outdir adapters_lc_s10 --mkey _s10 \
    || { echo "TRAIN $m FAILED"; continue; }
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
