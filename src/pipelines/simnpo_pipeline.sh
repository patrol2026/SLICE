#!/bin/bash
# SimNPO baseline: train + eval per epoch on Qwen (main splits) then DeepSeek.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True

all_zero () {  # tag -> exit 0 if forget+heldout+retain all 0
  python3 - "$1" <<'PY'
import json, sys
s = json.load(open("lc_summary.json")).get(sys.argv[1])
sys.exit(0 if s and sum(v["passed"] for v in s.values()) == 0 else 1)
PY
}

echo "=== TRAIN simnpo (qwen) ==="
python3 -u unlearn_lc.py --method simnpo || { echo "TRAIN simnpo FAILED"; exit 1; }
for ep in 1 2 3 4 5; do
  echo "=== EVAL simnpo epoch $ep ==="
  python3 -u eval_lc.py --tag simnpo_ep${ep} --batch-size 10 \
    --adapter adapters_lc/simnpo/epoch$ep \
    || { echo "EVAL simnpo_ep${ep} FAILED"; continue; }
  if all_zero "simnpo_ep${ep}"; then
    echo "=== simnpo epoch $ep is 0/0/0 — skipping remaining epochs ==="
    break
  fi
done

echo "=== TRAIN simnpo (ds) ==="
export LC_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
python3 -u unlearn_lc.py --method simnpo --splits leetcode_splits_ds.json \
  --outdir adapters_lc_ds --mkey _ds \
  || { echo "TRAIN simnpo (ds) FAILED"; exit 1; }
for ep in 1 2 3 4 5; do
  echo "=== EVAL simnpo_ds epoch $ep ==="
  python3 -u eval_lc.py --tag simnpo_ds_ep${ep} --batch-size 10 \
    --adapter adapters_lc_ds/simnpo/epoch$ep --splits leetcode_splits_ds.json \
    || { echo "EVAL simnpo_ds_ep${ep} FAILED"; continue; }
  if all_zero "simnpo_ds_ep${ep}"; then
    echo "=== simnpo_ds epoch $ep is 0/0/0 — skipping remaining epochs ==="
    break
  fi
done

echo "=== SIMNPO SUMMARY ==="
python3 - <<'PY'
import json
s = json.load(open('lc_summary.json'))
for tag in sorted(t for t in s if t.startswith('simnpo')):
    v = s[tag]
    print(tag, {k: f"{d['passed']}/{d['total']}" for k, d in v.items()})
PY
echo "=== DONE ==="
