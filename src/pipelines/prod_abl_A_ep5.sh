#!/bin/bash
# Protocol-A: finish the two missing ablations (noguard, nonll), EVAL EP5 ONLY.
# GPU-wait guard against co-tenant jobs. Idempotent.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PROD_SPLITS=prod_splits_A.json
export PROD_DATA=prod_slice_data_A.json
export PROD_RESULTS=prod_results_A.json
wait_gpu () { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 3000 ]; do sleep 180; done; }
have () { python3 -c "import json;assert '$1' in json.load(open('prod_results_A.json'))" 2>/dev/null; }
for ab in noguard nonll; do
  if [ ! -d adapters_prodA/slice_${ab}/epoch5 ]; then
    wait_gpu
    PROD_SLICE_OUT=adapters_prodA/slice_${ab} python3 -u prod_unlearn.py --ablation $ab \
      || { echo "ABL_A $ab TRAIN FAILED"; continue; }
  fi
  if have slice-${ab}_ep5; then echo "slice-${ab}_ep5 already in results, skip"; continue; fi
  wait_gpu
  python3 -u prod_eval.py --tag slice-${ab}_ep5 --adapter adapters_prodA/slice_${ab}/epoch5 \
    || echo "EVAL_A slice-${ab}_ep5 FAILED"
done
echo PROD_ABL_A_EP5_DONE
