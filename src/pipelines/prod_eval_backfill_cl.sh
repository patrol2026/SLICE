#!/bin/bash
# Backfill the CodeLlama epoch-1/epoch-3 evaluations skipped during the main
# campaign (run_prod_codellama.sbatch sets PROD_EVAL_EPOCHS=5). Uses the saved
# adapters_prod_cl_{B,A}/*/epoch{1,3}. Idempotent: rows already present in
# prod_results_cl_{B,A}.json are skipped. Needs a GPU, so on minsky:
#
#   sbatch --gres=gpu:1 --cpus-per-task=8 --mem=96G --time=08:00:00 \
#     --wrap 'export PATH=$PWD/.venv/bin:$PATH USE_TF=0 HF_HUB_OFFLINE=1; bash prod_eval_backfill_cl.sh' \
#     --output=slurm_backfill_cl_%j.out
#
# ~20 min per eval; B+A together is ~40 evals (~13 h), so expect 2 jobs.
# Override epochs with EPOCHS="1" or EPOCHS="3".
cd "$(dirname "$0")"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PROD_MODEL="codellama/CodeLlama-7b-Instruct-hf"
export PROD_LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
export PROD_MEM="adapters_prod_cl/memorized/epoch10"
export PROD_MEMCENSUS="prod_memorization_cl.jsonl"
export LC_MODEL="$PROD_MODEL" LC_GEN_BATCH=16
EPOCHS=${EPOCHS:-1 3}

have () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('$2')) else 1)" 2>/dev/null; }

for T in B A; do
  export PROD_SPLITS=prod_splits_cl_${T}.json PROD_DATA=prod_cil_data_cl_${T}.json
  export PROD_RESULTS=prod_results_cl_${T}.json PROD_ADIR=adapters_prod_cl_${T}
  # tag-prefix:adapter-dir pairs, matching run_prod_codellama.sh naming
  for pair in cil:cil prod:prod ga:ga gd:gd dpo:dpo npo:npo simnpo:simnpo ila:ila \
              cil-noil:cil_noil cil-noforget:cil_noforget cil-noguard:cil_noguard cil-nonll:cil_nonll; do
    tag=${pair%%:*} dir=$PROD_ADIR/${pair#*:}
    for ep in $EPOCHS; do
      [ -d $dir/epoch$ep ] || { echo "MISSING $dir/epoch$ep"; continue; }
      have ${tag}_ep${ep} "$PROD_RESULTS" || python3 -u prod_eval.py --tag ${tag}_ep${ep} --adapter $dir/epoch$ep
    done
  done
done
echo "=== BACKFILL DONE ==="
