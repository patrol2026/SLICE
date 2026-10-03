#!/bin/bash
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LC_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
export LC_RESULTS=results_leetcode_dsv2.jsonl
for ep in 3 4 5; do
  tag=cil_dsB_ep${ep}
  python3 -c "import json,sys;sys.exit(0 if '$tag' in json.load(open('lc_summary.json')) else 1)" && continue
  for try in 1 2 3 4 5; do
    while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -ge 6000 ]; do sleep 180; done
    python3 -u eval_lc.py --tag $tag --batch-size 10 \
      --adapter adapters_grid/dsB/cil/epoch$ep --splits splits_ds_B.json && break
    echo "retry $try for $tag"; sleep 300
  done
done
echo FINISH_DSB_DONE
