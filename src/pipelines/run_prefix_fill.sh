#!/bin/bash
# Full prefix-injection sweep for the still-empty cells.
#  - LeetCode: 3 models x both splits x {base + 8 methods} (Code Llama base/SLICE
#    already done, so only its baselines run). pass@1 -> LC_RESULTS_DIR/prefix_results.json
#  - Copyrighted (Qwen): split A all 9, split B remaining 6. BLEU -> prod_attack_results.json
# Epoch convention matches run_relearn_fill: prod->epoch3, memorized->epoch10,
# everything else ->epoch5. Idempotent (skips existing tags), serialized, GPU-guarded.
cd /path/to/slice || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LC_GEN_BATCH=32   # 7B models; DeepSeek section lowers this to 16
wait_gpu () { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 8000 ]; do sleep 120; done; }
BASE="ga gd npo dpo simnpo codeeraser prod"

adp () { # cell method -> adapter path (grid, else legacy)
  if [ -d adapters_grid/$1/$2/epoch5 ]; then echo adapters_grid/$1/$2/epoch5
  elif [ $1 = qwenB ] && [ -d adapters_lc/$2/epoch5 ]; then echo adapters_lc/$2/epoch5
  elif [ $1 = dsA ] && [ -d adapters_lc_ds/$2/epoch5 ]; then echo adapters_lc_ds/$2/epoch5
  fi
}
phave () { python3 -c "import json,sys,os;p=os.path.join('${LC_RESULTS_DIR:-.}','prefix_results.json');sys.exit(0 if os.path.exists(p) and '$1' in json.load(open(p)) else 1)"; }

run_leet () { # cell splits
  export LC_SPLITS=$2
  # base (pre-unlearn) once per cell
  tag=prefix_base_${1}
  if phave $tag; then echo "skip $tag"; else
    wait_gpu; echo "=== $(date +%H:%M) LEET $tag (base) ==="
    python3 -u prefix_lc.py --tag $tag || echo "FAIL $tag"
  fi
  for m in $BASE slice; do
    tag=prefix_${m}_${1}
    phave $tag && { echo "skip $tag"; continue; }
    a=$(adp $1 $m); [ -z "$a" ] && { echo "no adapter $1/$m"; continue; }
    wait_gpu; echo "=== $(date +%H:%M) LEET $tag ($a) ==="
    python3 -u prefix_lc.py --tag $tag --adapter $a || echo "FAIL $tag"
  done
}

# ---------------- LeetCode ----------------
unset LC_MODEL LC_RESULTS LC_LORA_TARGETS; export LC_RESULTS_DIR="."
export LC_RESULTS=results_leetcode.jsonl
run_leet qwenA splits_qwen_A.json
run_leet qwenB leetcode_splits.json
export LC_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
export LC_RESULTS=results_leetcode_dsv2.jsonl
export LC_GEN_BATCH=16   # 16B model, ~31GB weights; keep batch conservative
run_leet dsA leetcode_splits_ds.json
run_leet dsB splits_ds_B.json
export LC_MODEL=codellama/CodeLlama-7b-Instruct-hf
export LC_RESULTS=results_leetcode_codellama.jsonl
export LC_RESULTS_DIR=results_codellama
export LC_GEN_BATCH=32   # back to 7B batch
run_leet clA splits_cl_A.json
run_leet clB splits_cl_B.json

# ---------------- Copyrighted (Qwen) ----------------
unset LC_MODEL LC_RESULTS LC_LORA_TARGETS; export LC_RESULTS_DIR="."
export LC_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
MEM=adapters_prod/memorized/epoch10
cphave () { python3 -c "import json,sys;sys.exit(0 if 'prefix_$1' in json.load(open('prod_attack_results.json')) else 1)"; }
epof () { [ "$1" = prod ] && echo epoch3 || echo epoch5; }

# split A: base(memorized) + all 8 methods, suffixed _A
export PROD_SPLITS=prod_splits_A.json
cphave memorized_A || { wait_gpu; echo "=== $(date +%H:%M) COPY prefix_memorized_A ==="
  python3 -u prod_attacks.py prefix --tag memorized_A --adapters $MEM || echo "FAIL prefix_memorized_A"; }
for m in ga gd npo dpo simnpo codeeraser prod slice; do
  cphave ${m}_A && { echo "skip prefix_${m}_A"; continue; }
  wait_gpu; echo "=== $(date +%H:%M) COPY prefix_${m}_A ==="
  python3 -u prod_attacks.py prefix --tag ${m}_A --adapters $MEM,adapters_prodA/$m/$(epof $m) || echo "FAIL prefix_${m}_A"
done

# split B: remaining 6 baselines (memorized/slice/prod already present, unsuffixed)
export PROD_SPLITS=prod_splits.json
for m in ga gd npo dpo simnpo codeeraser; do
  cphave ${m}_B && { echo "skip prefix_${m}_B"; continue; }
  wait_gpu; echo "=== $(date +%H:%M) COPY prefix_${m}_B ==="
  python3 -u prod_attacks.py prefix --tag ${m}_B --adapters $MEM,adapters_prod/$m/$(epof $m) || echo "FAIL prefix_${m}_B"
done

echo "=== PREFIX FILL DONE $(date) ==="
