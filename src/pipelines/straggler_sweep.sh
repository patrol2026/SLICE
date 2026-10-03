#!/bin/bash
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LC_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
export LC_RESULTS=results_leetcode_dsv2.jsonl
export LC_GEN_BATCH=12
wait_gpu () { while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -ge 2500 ]; do sleep 120; done; }

wait_gpu
python3 -u eval_lc.py --tag slice-nonll_dsA_ep5 --batch-size 12 \
  --adapter adapters_ab/dsA_nonll/slice/epoch5 --splits leetcode_splits_ds.json || echo FAIL1
for ep in 1 3 5; do
  wait_gpu
  python3 -u eval_lc.py --tag slice-nonll_dsB_ep${ep} --batch-size 12 \
    --adapter adapters_ab/dsB_nonll/slice/epoch$ep --splits splits_ds_B.json || echo FAIL_ep$ep
done
wait_gpu
LC_PARAS=lc_paraphrases_dsB.json LC_SPLITS=splits_ds_B.json \
  python3 -u paraphrase_probe.py --tag para_slice_dsB --batch-size 12 \
  --adapter adapters_grid/dsB/slice/epoch5 || echo FAIL_para
wait_gpu
LC_SPLITS=splits_ds_B.json python3 -u relearn_lc.py --tag relearn_slice_dsB \
  --adapter adapters_grid/dsB/slice/epoch5 || echo FAIL_relearn
python3 mia_lc.py --auc
echo "=== SWEEP DONE ==="
