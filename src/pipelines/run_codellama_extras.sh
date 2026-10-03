#!/bin/bash
# CodeLlama extras: ablations (ep5) + attacks (MIA, relearn, prefix) + utility,
# BOTH splits. Writes everything under results_codellama/ (no overwrite of the
# Qwen/DS result files). Waits for the main grid to finish. Idempotent, GPU-guarded.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LC_MODEL=codellama/CodeLlama-7b-Instruct-hf
export LC_RESULTS=results_leetcode_codellama.jsonl
export LC_LORA_TARGETS=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
export LC_FORGET_ANN=ann_full_cl.jsonl
export LC_RESULTS_DIR=results_codellama
export LC_GEN_BATCH=10
mkdir -p "$LC_RESULTS_DIR"

wait_gpu () { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 8000 ]; do sleep 120; done; }
have () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('$LC_RESULTS_DIR/$2')) else 1)" 2>/dev/null; }

# wait for the main grid
while ! grep -q "CODELLAMA GRID DONE" nohup_codellama_grid.out 2>/dev/null; do
  echo "extras: waiting for main grid ($(date '+%T'))"; sleep 180; done

# utility of the base model (once, model-level)
have base_cl utility_results.json || { wait_gpu; python3 -u util_suite.py --tag base_cl || echo "UTIL base FAILED"; }

for cs in "clA splits_cl_A.json" "clB splits_cl_B.json"; do
  set -- $cs; cell=$1; splits=$2
  export LC_SPLITS=$splits
  export LC_MUTANTS=lc_mutants_${cell}.json
  CIL=adapters_grid/${cell}/cil/epoch5

  # ---- ablations (train + eval EP5 only) ----
  for ab in noil noforget noguard nonll; do
    outdir=adapters_ab/${cell}_${ab}
    [ -d $outdir/cil/epoch5 ] || { wait_gpu
      python3 -u unlearn_lc.py --method cil --ablation $ab --splits $splits \
        --outdir $outdir --mkey _${cell}_${ab} || { echo "ABL $ab/$cell FAILED"; continue; }; }
    tag=cil-${ab}_${cell}_ep5
    have $tag lc_summary.json || { wait_gpu
      python3 -u eval_lc.py --tag $tag --batch-size 10 --adapter $outdir/cil/epoch5 \
        --splits $splits || echo "EVAL $tag FAILED"; }
  done

  # ---- utility of the SLICE model ----
  have cil_${cell} utility_results.json || { wait_gpu
    python3 -u util_suite.py --tag cil_${cell} --adapter $CIL || echo "UTIL cil_$cell FAILED"; }

  # ---- MIA (base ref + SLICE) ----
  have base_${cell} mia_scores.json || { wait_gpu
    python3 -u mia_lc.py --tag base_${cell} || echo "MIA base_$cell FAILED"; }
  have cil_${cell}_ep5 mia_scores.json || { wait_gpu
    python3 -u mia_lc.py --tag cil_${cell}_ep5 --adapter $CIL || echo "MIA cil_$cell FAILED"; }

  # ---- relearning ----
  have relearn_cil_${cell} relearn_results.json || { wait_gpu
    python3 -u relearn_lc.py --tag relearn_cil_${cell} --adapter $CIL || echo "RELEARN $cell FAILED"; }

  # ---- prefix-injection (base + SLICE) ----
  have prefix_base_${cell} prefix_results.json || { wait_gpu
    python3 -u prefix_lc.py --tag prefix_base_${cell} || echo "PREFIX base_$cell FAILED"; }
  have prefix_cil_${cell} prefix_results.json || { wait_gpu
    python3 -u prefix_lc.py --tag prefix_cil_${cell} --adapter $CIL || echo "PREFIX cil_$cell FAILED"; }
done

echo "=== MIA AUC (CodeLlama) ==="; python3 mia_lc.py --auc || true
echo "=== CODELLAMA EXTRAS DONE ($(date '+%F %T')) ==="
