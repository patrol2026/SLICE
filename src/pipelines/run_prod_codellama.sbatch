#!/bin/bash
# Slurm wrapper for run_prod_codellama.sh on minsky (GPUs are only reachable
# inside a Slurm allocation; QOS gpu-limited = 1 GPU, 8h wall).
# The driver is idempotent, so on hitting the time limit this job resubmits
# itself and the next link resumes. A real failure (non-timeout exit) stops
# the chain so it can be diagnosed.
#
#   sbatch run_prod_codellama.sbatch
#
#SBATCH --job-name=prod-cl
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_prod_cl_%j.out

cd /path/to/slice
export PATH="$PWD/.venv/bin:$PATH" USE_TF=0 HF_HUB_OFFLINE=1
export PROD_EVAL_EPOCHS="5"   # ep1/ep3 adapters are saved; backfill later with prod_eval_backfill_cl.sh
LINK=${PROD_CL_LINK:-1}; MAXLINK=20
echo "[$(date)] link $LINK job $SLURM_JOB_ID on $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> nohup_prod_cl.out

timeout --signal=TERM --kill-after=120 465m bash -c 'bash run_prod_codellama.sh && bash prod_attacks_cl.sh && bash prod_cl_extras.sh' >> nohup_prod_cl.out 2>&1
rc=$?
echo "[$(date)] link $LINK driver exit $rc" >> nohup_prod_cl.out

util_ok () { python3 -c "import json,sys;u=json.load(open('utility_results.json'));sys.exit(0 if all('prodtask_cl_'+t in u for t in ('memorized','cil','prod')) else 1)" 2>/dev/null; }
if grep -q "=== PROD CODELLAMA EXTRAS COMPLETE ===" nohup_prod_cl.out && util_ok; then
  echo "campaign complete" >> nohup_prod_cl.out
elif [ $rc -eq 124 ] && [ $LINK -lt $MAXLINK ]; then
  PROD_CL_LINK=$((LINK+1)) sbatch --export=ALL,PROD_CL_LINK=$((LINK+1)) run_prod_codellama.sbatch >> nohup_prod_cl.out 2>&1
else
  echo "CHAIN STOPPED (rc=$rc, link=$LINK) - needs attention" >> nohup_prod_cl.out
fi
