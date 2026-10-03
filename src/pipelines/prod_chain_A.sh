#!/bin/bash
# Protocol A (10/20/70) campaign: waits for the B campaign, then runs
# SLICE + PROD + 6 baselines on the A splits. Results -> prod_results_A.json.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PROD_SPLITS=prod_splits_A.json
export PROD_DATA=prod_slice_data_A.json
export PROD_RESULTS=prod_results_A.json
export PROD_ADIR=adapters_prodA
export PROD_SLICE_OUT=adapters_prodA/slice

while ! grep -q PROD_BASELINES_DONE nohup_prod_baselines.out 2>/dev/null; do sleep 300; done

have () { python3 -c "import json;assert '$1' in json.load(open('$PROD_RESULTS'))" 2>/dev/null; }
collapsed () {
  python3 - "$1" <<'PY'
import json, os, sys
r = json.load(open(os.environ["PROD_RESULTS"])).get(sys.argv[1])
sys.exit(0 if r and all(v["bleu"] < 0.02 for v in r.values()) else 1)
PY
}

have memorized || python3 -u prod_eval.py --tag memorized
if [ ! -d adapters_prodA/slice/epoch5 ]; then
  python3 -u prod_unlearn.py || echo SLICE_A_TRAIN_FAILED
fi
for ep in 1 3 5; do
  have slice_ep${ep} || python3 -u prod_eval.py --tag slice_ep${ep} --adapter adapters_prodA/slice/epoch${ep}
done
for m in prod ga gd dpo npo simnpo codeeraser; do
  if [ ! -d adapters_prodA/$m/epoch5 ]; then
    python3 -u prod_baselines.py --method $m || { echo "TRAIN_A $m FAILED"; continue; }
  fi
  for ep in 1 3 5; do
    tag=${m}_ep${ep}
    have $tag && continue
    python3 -u prod_eval.py --tag $tag --adapter adapters_prodA/$m/epoch$ep || echo "EVAL_A $tag FAILED"
    if collapsed "$tag"; then echo "=== A: $m collapsed at ep$ep ==="; break; fi
  done
done
echo PROD_A_DONE
