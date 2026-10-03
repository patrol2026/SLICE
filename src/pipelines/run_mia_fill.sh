#!/bin/bash
# MIA fill for the still-empty cells:
#  - LeetCode: 7 baselines x 3 models x both splits (SLICE + base refs already on disk)
#  - Copyrighted (Qwen): baselines both splits (memorized/slice/prod already on disk for B)
# MIA is forward-pass only (no generation), so it is fast. Idempotent, GPU-guarded.
cd /path/to/slice || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
wait_gpu () { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 8000 ]; do sleep 60; done; }
BASE="ga gd npo dpo simnpo codeeraser prod"

adp () { # cell method -> adapter path (grid, else legacy)
  if [ -d adapters_grid/$1/$2/epoch5 ]; then echo adapters_grid/$1/$2/epoch5
  elif [ $1 = qwenB ] && [ -d adapters_lc/$2/epoch5 ]; then echo adapters_lc/$2/epoch5
  elif [ $1 = dsA ] && [ -d adapters_lc_ds/$2/epoch5 ]; then echo adapters_lc_ds/$2/epoch5
  fi
}
mhave () { python3 -c "import json,os,sys;p=os.path.join('${LC_RESULTS_DIR:-.}','mia_scores.json');sys.exit(0 if os.path.exists(p) and '$1' in json.load(open(p)) else 1)"; }

run_leet_mia () { # cell splits
  export LC_SPLITS=$2
  mhave base_${1} || { wait_gpu; echo "=== $(date +%H:%M) MIA base_${1} ==="; python3 -u mia_lc.py --tag base_${1} || echo "FAIL base_${1}"; }
  for m in $BASE; do
    tag=${m}_${1}_ep5
    mhave $tag && { echo "skip $tag"; continue; }
    a=$(adp $1 $m); [ -z "$a" ] && { echo "no adapter $1/$m"; continue; }
    wait_gpu; echo "=== $(date +%H:%M) MIA $tag ($a) ==="
    python3 -u mia_lc.py --tag $tag --adapter $a || echo "FAIL $tag"
  done
  python3 -u mia_lc.py --auc >/dev/null && echo "auc updated ($LC_RESULTS_DIR)"
}

# ---------- LeetCode: Qwen + DeepSeek (RDIR=.) ----------
unset LC_MODEL LC_RESULTS LC_LORA_TARGETS; export LC_RESULTS_DIR="."
export LC_RESULTS=results_leetcode.jsonl
run_leet_mia qwenA splits_qwen_A.json
run_leet_mia qwenB leetcode_splits.json
export LC_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
export LC_RESULTS=results_leetcode_dsv2.jsonl
run_leet_mia dsA leetcode_splits_ds.json
run_leet_mia dsB splits_ds_B.json
# ---------- LeetCode: Code Llama (RDIR=results_codellama) ----------
export LC_MODEL=codellama/CodeLlama-7b-Instruct-hf
export LC_RESULTS=results_leetcode_codellama.jsonl
export LC_RESULTS_DIR=results_codellama
run_leet_mia clA splits_cl_A.json
run_leet_mia clB splits_cl_B.json

# ---------- Copyrighted (Qwen) MIA ----------
unset LC_MODEL LC_RESULTS LC_LORA_TARGETS; export LC_RESULTS_DIR="."
export LC_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
MEM=adapters_prod/memorized/epoch10
chave () { python3 -c "import json,sys;sys.exit(0 if 'mia_$1' in json.load(open('prod_attack_results.json')) else 1)"; }
epof () { [ "$1" = prod ] && echo epoch3 || echo epoch5; }

# split A: base + all 8 methods, suffixed _A
export PROD_SPLITS=prod_splits_A.json
chave memorized_A || { wait_gpu; echo "=== $(date +%H:%M) COPY mia_memorized_A ==="; python3 -u prod_attacks.py mia --tag memorized_A --adapters $MEM || echo "FAIL mia_memorized_A"; }
for m in ga gd npo dpo simnpo codeeraser prod slice; do
  chave ${m}_A && { echo "skip mia_${m}_A"; continue; }
  wait_gpu; echo "=== $(date +%H:%M) COPY mia_${m}_A ==="
  python3 -u prod_attacks.py mia --tag ${m}_A --adapters $MEM,adapters_prodA/$m/$(epof $m) || echo "FAIL mia_${m}_A"
done
# split B: base+slice+prod already present (unsuffixed); compute the 6 remaining baselines _B
export PROD_SPLITS=prod_splits.json
for m in ga gd npo dpo simnpo codeeraser; do
  chave ${m}_B && { echo "skip mia_${m}_B"; continue; }
  wait_gpu; echo "=== $(date +%H:%M) COPY mia_${m}_B ==="
  python3 -u prod_attacks.py mia --tag ${m}_B --adapters $MEM,adapters_prod/$m/$(epof $m) || echo "FAIL mia_${m}_B"
done

echo "=== MIA FILL DONE $(date) ==="
