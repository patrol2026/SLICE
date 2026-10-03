#!/bin/bash
# Fill missing LeetCode utility cells for all baseline methods that have an
# ep5 adapter in adapters_grid/{cell}/{method}/epoch5. Idempotent (skips tags
# already in utility_results.json). Serialized on one GPU via wait_gpu.
cd /path/to/slice || exit 1
wait_gpu () { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 8000 ]; do sleep 120; done; }
BASE="ga gd dpo npo simnpo codeeraser prod"

run_block () {   # $1=model  $2=results_dir  then: cells...
  export LC_MODEL="$1"; export LC_RESULTS_DIR="$2"; mkdir -p "$2"
  local rdir="$2"; shift 2
  for cell in "$@"; do
    for m in $BASE; do
      ad=adapters_grid/$cell/$m/epoch5
      [ -d "$ad" ] || continue
      tag=${m}_${cell}_ep5
      if python3 -c "import json,os,sys;p=os.path.join('$rdir','utility_results.json');sys.exit(0 if os.path.exists(p) and '$tag' in json.load(open(p)) else 1)"; then
        echo "skip $tag (done)"; continue
      fi
      wait_gpu
      echo "=== $(date +%H:%M) $tag  [$LC_MODEL] ==="
      python3 -u util_suite.py --tag "$tag" --adapter "$ad" || echo "UTIL $tag FAILED"
    done
  done
}

run_block "Qwen/Qwen2.5-Coder-7B-Instruct"            "."               qwenA qwenB
run_block "codellama/CodeLlama-7b-Instruct-hf"        "results_codellama" clA clB
run_block "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct" "."             dsA dsB
echo "=== ALL UTIL FILL DONE $(date) ==="
