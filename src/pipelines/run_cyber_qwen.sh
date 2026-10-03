#!/bin/bash
# CyberSecEval (insecure-code) unlearning for QWEN, both splits.
# memorize -> funnel -> prep -> train SLICE+baselines+ablations -> eval ep5.
# Verified subset (737 regex records). Results -> results_cyber_qwen/.
# Idempotent, GPU-guarded. Reuses prod_unlearn.py / prod_baselines.py.
cd "$(dirname "$0")"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PROD_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
export CY_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
export PROD_LORA_TARGETS=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
export CY_LORA_TARGETS=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
export PROD_MEM=adapters_cyber_qwen/memorized/epoch10
export CY_MEM=adapters_cyber_qwen/memorized/epoch10
export CY_MEM_OUT=adapters_cyber_qwen/memorized
export CY_RESULTS_DIR=results_cyber_qwen
mkdir -p "$CY_RESULTS_DIR"

wait_gpu () { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 8000 ]; do sleep 120; done; }
have () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('$CY_RESULTS_DIR/cyber_results_$2.json')) else 1)" 2>/dev/null; }
ev () { local tag=$1 ad=$2 s=$3; have $tag $s && { echo "eval $tag done"; return; }
        wait_gpu; python3 -u cyber_eval.py --tag $tag --adapter $ad --split $s || echo "EVAL $tag FAILED"; }

# 0. memorize (Qwen) - r=64, 10 epochs
[ -d adapters_cyber_qwen/memorized/epoch10 ] || { wait_gpu
  python3 -u cyber_memorize.py || exit 1; }

# 1. funnel + 2. prep
[ -f cyber_memorization.jsonl ] || { wait_gpu; python3 -u cyber_funnel.py || exit 1; }
[ -f cyber_splits_A.json ] || python3 -u cyber_prep.py || exit 1

# 3. train + eval per split
for s in A B; do
  export PROD_SPLITS=cyber_splits_${s}.json
  export PROD_DATA=cyber_slice_data_${s}.json
  export PROD_ADIR=adapters_cyber_qwen_${s}
  ev memorized_${s} adapters_cyber_qwen/memorized/epoch10 $s

  [ -d adapters_cyber_qwen_${s}/slice/epoch5 ] || { wait_gpu
    PROD_SLICE_OUT=adapters_cyber_qwen_${s}/slice python3 -u prod_unlearn.py || echo "TRAIN slice/$s FAILED"; }
  ev slice_${s}_ep5 adapters_cyber_qwen_${s}/slice/epoch5 $s

  for m in ga gd dpo npo simnpo codeeraser prod; do
    [ -d adapters_cyber_qwen_${s}/${m}/epoch5 ] || { wait_gpu
      python3 -u prod_baselines.py --method $m || { echo "TRAIN $m/$s FAILED"; continue; }; }
    ev ${m}_${s}_ep5 adapters_cyber_qwen_${s}/${m}/epoch5 $s
  done

  for ab in noil noforget noguard nonll; do
    [ -d adapters_cyber_qwen_${s}/slice_${ab}/epoch5 ] || { wait_gpu
      PROD_SLICE_OUT=adapters_cyber_qwen_${s}/slice_${ab} python3 -u prod_unlearn.py --ablation $ab \
        || { echo "TRAIN slice-${ab}/$s FAILED"; continue; }; }
    ev slice-${ab}_${s}_ep5 adapters_cyber_qwen_${s}/slice_${ab}/epoch5 $s
  done
done

echo "=== CYBER QWEN DONE ($(date '+%F %T')) ==="
python3 - <<'PY'
import json
for s in ("A","B"):
    d=json.load(open(f"results_cyber_qwen/cyber_results_{s}.json"))
    print(f"--- split {s} (forget/heldout/retain: bleu | analyzer_pass) ---")
    for tag in sorted(d):
        v=d[tag]
        print(f"  {tag:20s} " + " ".join(
            f"{k[:3]}:{v[k]['bleu']:.2f}/{v[k]['analyzer_pass']:.2f}" for k in ('forget','heldout','retain')))
PY
