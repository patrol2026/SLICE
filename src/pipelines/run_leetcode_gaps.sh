#!/bin/bash
# Fill remaining LeetCode utility gaps:
#  1. baseline utility on existing legacy adapters (Qwen split-B = adapters_lc,
#     DeepSeek split-A = adapters_lc_ds) -- NO retrain needed.
#  2. PROD-LeetCode: train (never existed) then utility, for Qwen A/B + DS A/B.
cd /path/to/slice || exit 1
# chain after the copyright split-A fill so we never contend for the GPU
while ! grep -q "CP-A FILL DONE" copy_fill_A.log 2>/dev/null; do sleep 60; done
wait_gpu () { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 8000 ]; do sleep 120; done; }
uhave () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('utility_results.json')) else 1)"; }
QWEN="Qwen/Qwen2.5-Coder-7B-Instruct"
DS="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
export LC_RESULTS_DIR="."

# ---- 1a. Qwen split-B baselines (adapters_lc = leetcode_splits.json = split B) ----
export LC_MODEL="$QWEN"
for m in ga gd npo dpo; do
  tag=${m}_qwenB_ep5
  uhave $tag && { echo "skip $tag"; continue; }
  wait_gpu; echo "=== $(date +%H:%M) UTIL $tag ==="
  python3 -u util_suite.py --tag $tag --adapter adapters_lc/$m/epoch5 || echo "FAIL $tag"
done
# ---- 1b. DeepSeek split-A baselines (adapters_lc_ds = leetcode_splits_ds.json = split A) ----
export LC_MODEL="$DS"
for m in ga gd npo dpo; do
  tag=${m}_dsA_ep5
  uhave $tag && { echo "skip $tag"; continue; }
  wait_gpu; echo "=== $(date +%H:%M) UTIL $tag ==="
  python3 -u util_suite.py --tag $tag --adapter adapters_lc_ds/$m/epoch5 || echo "FAIL $tag"
done

# ---- 2. PROD-LeetCode: train then utility ----
export PROD_TOP_P=0.8
train_prod () {  # cell splits mutants ann
  [ -d adapters_grid/$1/prod/epoch5 ] && { echo "prod/$1 exists"; return; }
  wait_gpu; echo "=== $(date +%H:%M) TRAIN prod/$1 ==="
  LC_MUTANTS=$3 LC_FORGET_ANN=$4 python3 -u unlearn_lc.py --method prod \
    --splits $2 --outdir adapters_grid/$1 --mkey _$1 || echo "TRAIN prod/$1 FAILED"
}
util_prod () {  # cell
  local tag=prod_${1}_ep5
  uhave $tag && { echo "skip $tag"; return; }
  [ -d adapters_grid/$1/prod/epoch5 ] || { echo "no prod/$1 adapter"; return; }
  wait_gpu; echo "=== $(date +%H:%M) UTIL $tag ==="
  python3 -u util_suite.py --tag $tag --adapter adapters_grid/$1/prod/epoch5 || echo "FAIL $tag"
}
# Qwen
export LC_MODEL="$QWEN"; unset LC_RESULTS LC_LORA_TARGETS
train_prod qwenA splits_qwen_A.json lc_mutants_qwenA.json ann_full_qwen.jsonl; util_prod qwenA
train_prod qwenB leetcode_splits.json lc_mutants_qwenB.json ann_full_qwen.jsonl; util_prod qwenB
# DeepSeek
export LC_MODEL="$DS"; export LC_RESULTS="results_leetcode_dsv2.jsonl"
export LC_LORA_TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
train_prod dsA leetcode_splits_ds.json lc_mutants_dsA.json ann_full_ds.jsonl; util_prod dsA
train_prod dsB splits_ds_B.json lc_mutants_dsB.json ann_full_ds.jsonl; util_prod dsB

echo "=== LEETCODE GAPS FILL DONE $(date) ==="
