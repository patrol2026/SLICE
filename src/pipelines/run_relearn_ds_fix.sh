#!/bin/bash
# Re-run the DeepSeek LeetCode relearn cells that OOM'd. Fixes: correct
# PYTORCH_CUDA_ALLOC_CONF var + smaller generation batch for the 16B MoE.
cd /path/to/slice || exit 1
export RELEARN_EPOCHS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LC_GEN_BATCH=8
export LC_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
export LC_RESULTS=results_leetcode_dsv2.jsonl
export LC_LORA_TARGETS=q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj
export LC_RESULTS_DIR="."
wait_gpu () { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 8000 ]; do sleep 120; done; }
rhave () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('relearn_results.json')) else 1)"; }
adp () {  # cell method
  if [ -d adapters_grid/$1/$2/epoch5 ]; then echo adapters_grid/$1/$2/epoch5
  elif [ $1 = dsA ] && [ -d adapters_lc_ds/$2/epoch5 ]; then echo adapters_lc_ds/$2/epoch5
  fi
}
run () {  # cell splits
  export LC_SPLITS=$2
  for m in ga gd npo dpo simnpo ila prod; do
    tag=relearn_${m}_${1}
    rhave $tag && { echo "skip $tag"; continue; }
    a=$(adp $1 $m); [ -z "$a" ] && { echo "no adapter $1/$m"; continue; }
    wait_gpu; echo "=== $(date +%H:%M) $tag ($a) ==="
    python3 -u relearn_lc.py --tag $tag --adapter $a || echo "FAIL $tag"
  done
}
run dsA leetcode_splits_ds.json
run dsB splits_ds_B.json
echo "=== DS RELEARN FIX DONE $(date) ==="
