#!/bin/bash
# Relearn attack for all methods EXCEPT SLICE: LeetCode (3 models x both splits,
# 7 baselines) + Copyrighted Qwen (split A all 8, split B all except SLICE).
# Epochs capped at 6 (spreadsheet needs 0/2/4/6). Idempotent, serialized.
cd /path/to/slice || exit 1
export RELEARN_EPOCHS=6
export PYTORCH_ALLOC_CONF=expandable_segments:True
wait_gpu () { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 8000 ]; do sleep 120; done; }
BASE="ga gd npo dpo simnpo codeeraser prod"

adp () { # cell method -> adapter path (grid, else legacy)
  if [ -d adapters_grid/$1/$2/epoch5 ]; then echo adapters_grid/$1/$2/epoch5
  elif [ $1 = qwenB ] && [ -d adapters_lc/$2/epoch5 ]; then echo adapters_lc/$2/epoch5
  elif [ $1 = dsA ] && [ -d adapters_lc_ds/$2/epoch5 ]; then echo adapters_lc_ds/$2/epoch5
  fi
}
rhave () { python3 -c "import json,sys,os;p=os.path.join('${LC_RESULTS_DIR:-.}','relearn_results.json');sys.exit(0 if os.path.exists(p) and '$1' in json.load(open(p)) else 1)"; }
run_leet () { # cell splits
  export LC_SPLITS=$2
  for m in $BASE; do
    tag=relearn_${m}_${1}
    rhave $tag && { echo "skip $tag"; continue; }
    a=$(adp $1 $m); [ -z "$a" ] && { echo "no adapter $1/$m"; continue; }
    wait_gpu; echo "=== $(date +%H:%M) LEET $tag ($a) ==="
    python3 -u relearn_lc.py --tag $tag --adapter $a || echo "FAIL $tag"
  done
}

# ---------- LeetCode ----------
unset LC_MODEL LC_RESULTS LC_LORA_TARGETS; export LC_RESULTS_DIR="."
run_leet qwenA splits_qwen_A.json
run_leet qwenB leetcode_splits.json
export LC_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
export LC_RESULTS=results_leetcode_dsv2.jsonl
export LC_LORA_TARGETS=q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj
run_leet dsA leetcode_splits_ds.json
run_leet dsB splits_ds_B.json
export LC_MODEL=codellama/CodeLlama-7b-Instruct-hf
export LC_RESULTS=results_leetcode_codellama.jsonl
export LC_RESULTS_DIR=results_codellama; unset LC_LORA_TARGETS
run_leet clA splits_cl_A.json
run_leet clB splits_cl_B.json

# ---------- Copyrighted (Qwen) ----------
unset LC_MODEL LC_RESULTS LC_LORA_TARGETS; export LC_RESULTS_DIR="."
MEM=adapters_prod/memorized/epoch10
cphave () { python3 -c "import json,sys;sys.exit(0 if 'relearn_$1' in json.load(open('prod_attack_results.json')) else 1)"; }
epof () { [ "$1" = prod ] && echo epoch3 || echo epoch5; }
export PROD_SPLITS=prod_splits_A.json          # split A: all 8 methods
for m in ga gd npo dpo simnpo codeeraser prod slice; do
  cphave ${m}_A && { echo "skip relearn_${m}_A"; continue; }
  wait_gpu; echo "=== $(date +%H:%M) COPY relearn_${m}_A ==="
  python3 -u prod_attacks.py relearn --tag ${m}_A --adapters $MEM,adapters_prodA/$m/$(epof $m) || echo "FAIL relearn_${m}_A"
done
export PROD_SPLITS=prod_splits.json            # split B: all except SLICE (slice done)
for m in ga gd npo dpo simnpo codeeraser prod; do
  cphave ${m}_B && { echo "skip relearn_${m}_B"; continue; }
  wait_gpu; echo "=== $(date +%H:%M) COPY relearn_${m}_B ==="
  python3 -u prod_attacks.py relearn --tag ${m}_B --adapters $MEM,adapters_prod/$m/$(epof $m) || echo "FAIL relearn_${m}_B"
done
echo "=== RELEARN FILL DONE $(date) ==="
