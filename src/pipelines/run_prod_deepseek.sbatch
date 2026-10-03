#!/bin/bash
# Slurm wrapper for run_prod_deepseek.sh on minsky (GPUs are only reachable
# inside a Slurm allocation; QOS gpu-limited = 1 GPU, 8h wall).
# The driver is idempotent, so on hitting the time limit this job resubmits
# itself and the next link resumes. A real failure (non-timeout exit) stops
# the chain so it can be diagnosed.
#
#   sbatch run_prod_deepseek.sbatch
#
#SBATCH --job-name=prod-ds
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_prod_ds_%j.out

cd /path/to/slice
export PATH="$PWD/.venv/bin:$PATH" USE_TF=0 HF_HUB_OFFLINE=1
export PROD_EVAL_EPOCHS="5"   # ep1/ep3 adapters are saved; backfill later with prod_eval_backfill_ds.sh
LINK=${PROD_DS_LINK:-1}; MAXLINK=20
echo "[$(date)] link $LINK job $SLURM_JOB_ID on $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> nohup_prod_ds.out

timeout --signal=TERM --kill-after=120 465m bash -c 'bash run_prod_deepseek.sh && bash prod_attacks_ds.sh && bash prod_ds_extras.sh' >> nohup_prod_ds.out 2>&1
rc=$?
echo "[$(date)] link $LINK driver exit $rc" >> nohup_prod_ds.out

util_ok () { python3 -c "import json,sys;u=json.load(open('utility_results.json'));sys.exit(0 if all('prodtask_ds_'+t in u for t in ('memorized','cil','prod')) else 1)" 2>/dev/null; }
if grep -q "=== PROD DEEPSEEK EXTRAS COMPLETE ===" nohup_prod_ds.out && util_ok; then
  echo "campaign complete" >> nohup_prod_ds.out
elif [ $rc -eq 124 ] && [ $LINK -lt $MAXLINK ]; then
  PROD_DS_LINK=$((LINK+1)) sbatch --export=ALL,PROD_DS_LINK=$((LINK+1)) run_prod_deepseek.sbatch >> nohup_prod_ds.out 2>&1
else
  echo "CHAIN STOPPED (rc=$rc, link=$LINK) - needs attention" >> nohup_prod_ds.out
fi
