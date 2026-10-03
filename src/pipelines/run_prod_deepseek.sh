#!/bin/bash
# ==========================================================================
# Full copyrighted-code campaign for DeepSeek-Coder-V2-Lite, BOTH splits.
# Run on minsky (or any box with a >=48GB GPU + the transferred folder).
#
#   cd ~/slice
#   nohup bash run_prod_deepseek.sh > nohup_prod_ds.out 2>&1 &
#
# Idempotent: re-running skips finished stages. GPU-guarded against co-tenants.
# ==========================================================================
cd "$(dirname "$0")"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ---- DeepSeek model config (MoE LoRA targets, as in the LeetCode DS runs) ----
export PROD_MODEL="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
export PROD_LORA_TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
export PROD_MEM="adapters_prod_ds/memorized/epoch10"
export PROD_MEM_OUT="adapters_prod_ds/memorized"
export PROD_MEMCENSUS="prod_memorization_ds.jsonl"
export PROD_FUNNEL_SUMMARY="prod_funnel_summary_ds.json"
export LC_MODEL="$PROD_MODEL"          # for util_suite (reads run_leetcode.MODEL)
export LC_GEN_BATCH=16

wait_gpu () { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 3000 ]; do sleep 180; done; }
have () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('$2')) else 1)" 2>/dev/null; }

# ---------- 1. memorize (10 epochs) ----------
if [ ! -d adapters_prod_ds/memorized/epoch10 ]; then
  echo "=== MEMORIZE (deepseek) ==="; wait_gpu
  python3 -u memorize_prod.py || { echo MEMORIZE_FAILED; exit 1; }
fi

# ---------- 2. funnel (census + threshold) ----------
if [ ! -f prod_memorization_ds.jsonl ]; then
  echo "=== FUNNEL (deepseek) ==="; wait_gpu
  python3 -u prod_funnel.py || { echo FUNNEL_FAILED; exit 1; }
fi
cp -n prod_memorization_ds.jsonl prod_memorization_ds_ep10.jsonl 2>/dev/null || true

# ---------- 3. per-split campaigns ----------
run_split () {   # $1=tag(B/A)  $2=ratioF  $3=ratioH
  local T=$1 RF=$2 RH=$3
  local SP=prod_splits_ds_${T}.json DT=prod_slice_data_ds_${T}.json
  local RES=prod_results_ds_${T}.json ADIR=adapters_prod_ds_${T}
  export PROD_SPLITS=$SP PROD_DATA=$DT PROD_RESULTS=$RES PROD_ADIR=$ADIR

  if [ ! -f "$DT" ]; then
    echo "=== PREP $T (ratios $RF/$RH) ==="
    PROD_RATIO_F=$RF PROD_RATIO_H=$RH PROD_SPLITS_OUT=$SP PROD_DATA_OUT=$DT \
      python3 -u prod_slice_prep.py || { echo "PREP $T FAILED"; return 1; }
  fi

  have memorized "$RES" || { wait_gpu; python3 -u prod_eval.py --tag memorized; }

  # SLICE (main)
  if [ ! -d $ADIR/slice/epoch5 ]; then
    wait_gpu; PROD_SLICE_OUT=$ADIR/slice python3 -u prod_unlearn.py || echo "SLICE $T FAILED"
  fi
  for ep in 1 3 5; do
    have slice_ep${ep} "$RES" || { wait_gpu; python3 -u prod_eval.py --tag slice_ep${ep} --adapter $ADIR/slice/epoch$ep; }
  done

  # baselines
  for m in prod ga gd dpo npo simnpo codeeraser; do
    if [ ! -d $ADIR/$m/epoch5 ]; then
      wait_gpu; python3 -u prod_baselines.py --method $m || { echo "TRAIN $m/$T FAILED"; continue; }
    fi
    for ep in 1 3 5; do
      have ${m}_ep${ep} "$RES" || { wait_gpu; python3 -u prod_eval.py --tag ${m}_ep${ep} --adapter $ADIR/$m/epoch$ep; }
    done
  done

  # ablations
  for ab in noil noforget noguard nonll; do
    if [ ! -d $ADIR/slice_${ab}/epoch5 ]; then
      wait_gpu; PROD_SLICE_OUT=$ADIR/slice_${ab} python3 -u prod_unlearn.py --ablation $ab || { echo "ABL $ab/$T FAILED"; continue; }
    fi
    for ep in 1 3 5; do
      have slice-${ab}_ep${ep} "$RES" || { wait_gpu; python3 -u prod_eval.py --tag slice-${ab}_ep${ep} --adapter $ADIR/slice_${ab}/epoch$ep; }
    done
  done
  echo "=== SPLIT $T DONE ==="
}

run_split B 0.2 0.2
run_split A 0.1 0.2

# ---------- 4. utility (state-based, protocol B adapters) ----------
MEMDS=adapters_prod_ds/memorized/epoch10
have prodtask_ds_memorized utility_results.json || { wait_gpu; python3 -u util_suite.py --tag prodtask_ds_memorized --adapter $MEMDS; }
have prodtask_ds_slice utility_results.json       || { wait_gpu; python3 -u util_suite.py --tag prodtask_ds_slice --adapter $MEMDS,adapters_prod_ds_B/slice/epoch5; }
have prodtask_ds_prod utility_results.json      || { wait_gpu; python3 -u util_suite.py --tag prodtask_ds_prod --adapter $MEMDS,adapters_prod_ds_B/prod/epoch3; }

echo "=== PROD DEEPSEEK CAMPAIGN COMPLETE ==="
