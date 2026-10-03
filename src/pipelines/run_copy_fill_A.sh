#!/bin/bash
cd /path/to/slice || exit 1
# wait for the CP-B fill to finish so we don't contend for the GPU
while ! grep -q "CP-B FILL DONE" copy_fill.log 2>/dev/null; do sleep 60; done
export LC_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
export LC_RESULTS_DIR="."
MEM=adapters_prod/memorized/epoch10
wait_gpu () { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 8000 ]; do sleep 120; done; }
for m in ga gd npo dpo simnpo codeeraser; do
  tag=prodtask_${m}_A
  python3 -c "import json,sys;sys.exit(0 if '$tag' in json.load(open('utility_results.json')) else 1)" && { echo "skip $tag"; continue; }
  wait_gpu
  echo "=== $(date +%H:%M) $tag ==="
  python3 -u util_suite.py --tag "$tag" --adapter "$MEM,adapters_prodA/$m/epoch5" || echo "FAIL $tag"
done
echo "=== COPYRIGHT CP-A FILL DONE $(date) ==="
