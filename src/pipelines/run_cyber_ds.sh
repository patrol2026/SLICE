#!/bin/bash
# CyberSecEval unlearning for DeepSeek-Coder-V2-Lite (16B MoE), both splits.
# memorize -> funnel -> prep -> train SLICE+baselines+ablations -> eval ep5.
# Verified 737-record subset. Results -> results_cyber_ds/. Idempotent, GPU-guarded.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PROD_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
export CY_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
export PROD_LORA_TARGETS=q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj
export CY_LORA_TARGETS=q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj
export CY_MEM_OUT=adapters_cyber_ds/memorized
export CY_MEM=adapters_cyber_ds/memorized/epoch10
export PROD_MEM=adapters_cyber_ds/memorized/epoch10
export CY_MEMCENSUS=cyber_memorization_ds.jsonl
export CY_TAG=_ds
export CY_RESULTS_DIR=results_cyber_ds
mkdir -p "$CY_RESULTS_DIR"

wait_gpu () { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 8000 ]; do sleep 120; done; }
have () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('$CY_RESULTS_DIR/cyber_results_$2.json')) else 1)" 2>/dev/null; }
ev () { local tag=$1 ad=$2 s=$3; have $tag $s && { echo "eval $tag done"; return; }
        wait_gpu; python3 -u cyber_eval.py --tag $tag --adapter $ad --split $s || echo "EVAL $tag FAILED"; }

# 0. memorize (DeepSeek) r=64, 10 epochs
[ -d adapters_cyber_ds/memorized/epoch10 ] || { wait_gpu; python3 -u cyber_memorize.py || exit 1; }
# 1. funnel + 2. prep
[ -f cyber_memorization_ds.jsonl ] || { wait_gpu; python3 -u cyber_funnel.py || exit 1; }
[ -f cyber_splits_ds_A.json ] || python3 -u cyber_prep.py || exit 1

# 3. train + eval per split
for s in A B; do
  export PROD_SPLITS=cyber_splits_ds_${s}.json
  export PROD_DATA=cyber_slice_data_ds_${s}.json
  export PROD_ADIR=adapters_cyber_ds_${s}
  ev memorized_${s} adapters_cyber_ds/memorized/epoch10 $s

  [ -d adapters_cyber_ds_${s}/slice/epoch5 ] || { wait_gpu
    PROD_SLICE_OUT=adapters_cyber_ds_${s}/slice python3 -u prod_unlearn.py || echo "TRAIN slice/$s FAILED"; }
  ev slice_${s}_ep5 adapters_cyber_ds_${s}/slice/epoch5 $s

  for m in ga gd dpo npo simnpo codeeraser prod; do
    [ -d adapters_cyber_ds_${s}/${m}/epoch5 ] || { wait_gpu
      python3 -u prod_baselines.py --method $m || { echo "TRAIN $m/$s FAILED"; continue; }; }
    ev ${m}_${s}_ep5 adapters_cyber_ds_${s}/${m}/epoch5 $s
  done

  for ab in noil noforget noguard nonll; do
    [ -d adapters_cyber_ds_${s}/slice_${ab}/epoch5 ] || { wait_gpu
      PROD_SLICE_OUT=adapters_cyber_ds_${s}/slice_${ab} python3 -u prod_unlearn.py --ablation $ab \
        || { echo "TRAIN slice-${ab}/$s FAILED"; continue; }; }
    ev slice-${ab}_${s}_ep5 adapters_cyber_ds_${s}/slice_${ab}/epoch5 $s
  done
done

echo "=== CYBER DEEPSEEK DONE ($(date '+%F %T')) ==="
