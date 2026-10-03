#!/bin/bash
# Protocol-A ablations, with GPU-wait guard against co-tenant jobs.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PROD_SPLITS=prod_splits_A.json
export PROD_DATA=prod_slice_data_A.json
export PROD_RESULTS=prod_results_A.json
wait_gpu () { while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -ge 3000 ]; do sleep 180; done; }
have () { python3 -c "import json;assert '$1' in json.load(open('prod_results_A.json'))" 2>/dev/null; }
for ab in noil noforget noguard nonll; do
  if [ ! -d adapters_prodA/slice_${ab}/epoch5 ]; then
    wait_gpu
    PROD_SLICE_OUT=adapters_prodA/slice_${ab} python3 -u prod_unlearn.py --ablation $ab \
      || { echo "ABL_A $ab FAILED"; continue; }
  fi
  for ep in 1 3 5; do
    have slice-${ab}_ep${ep} && continue
    wait_gpu
    python3 -u prod_eval.py --tag slice-${ab}_ep${ep} --adapter adapters_prodA/slice_${ab}/epoch${ep} \
      || echo "EVAL_A slice-${ab}_ep${ep} FAILED"
  done
done
echo PROD_ABL_A_DONE
