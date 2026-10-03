#!/bin/bash
# DeepSeek-Coder-V2-Lite run: all methods, per-epoch evals, 0/0/0 skip rule.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LC_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
export LC_RESULTS=results_leetcode_dsv2.jsonl
export LC_FORGET_ANN=leetcode_forget_ds_important_lines.jsonl
export LC_MUTANTS=lc_mutants_ds.json
export LC_LORA_TARGETS=q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj
SPLITS=leetcode_splits_ds.json

all_zero() {
  python - "$1" <<'PY'
import json, sys
s = json.load(open("lc_summary.json")).get(sys.argv[1])
sys.exit(0 if s and sum(v["passed"] for v in s.values()) == 0 else 1)
PY
}

for m in ga gd dpo npo ila cil; do
  if [ -d adapters_lc_ds/$m/epoch5 ]; then
    echo "=== TRAIN $m (ds) already done, skipping ==="
  else
    echo "=== TRAIN $m (ds) ==="
    python unlearn_lc.py --method $m --splits $SPLITS \
      --outdir adapters_lc_ds --mkey _ds \
      || { echo "TRAIN $m FAILED"; continue; }
  fi
  for ep in 1 2 3 4 5; do
    echo "=== EVAL ${m}_ds epoch $ep ==="
    python eval_lc.py --tag ${m}_ds_ep${ep} --batch-size 10 \
      --adapter adapters_lc_ds/$m/epoch$ep --splits $SPLITS \
      || { echo "EVAL ${m}_ds_ep${ep} FAILED"; continue; }
    if all_zero "${m}_ds_ep${ep}"; then
      echo "=== ${m}_ds epoch $ep is 0/0/0 — skipping remaining epochs ==="
      break
    fi
  done
done

echo "=== DS RUN DONE ==="
cat lc_summary.json | python -c "import json,sys; s=json.load(sys.stdin); print(json.dumps({k:v for k,v in s.items() if '_ds' in k}, indent=1))"
