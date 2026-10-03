#!/bin/bash
# Slurm wrapper for run_cyber_qwen.sh on minsky (GPUs are only reachable inside
# an allocation; QOS gpu-limited = 1 GPU, 8h wall). The driver is idempotent, so
# on hitting the time limit this job resubmits itself and the next link resumes.
# A real failure (non-timeout exit) stops the chain for diagnosis.
#
#   cd ~/slice/cybersec && sbatch run_cyber_qwen.sbatch
#
#SBATCH --job-name=cyber-qwen
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_cyber_qwen_%j.out

cd /path/to/slice/cybersec
export PATH="/path/to/slice/.venv/bin:$PATH" USE_TF=0 HF_HUB_OFFLINE=1
LINK=${CY_LINK:-1}; MAXLINK=20
echo "[$(date)] link $LINK job $SLURM_JOB_ID on $(nvidia-smi --query-gpu=name --format=csv,noheader)" >> nohup_cyber_qwen.out

timeout --signal=TERM --kill-after=120 465m bash -c 'bash run_cyber_qwen.sh && bash run_cyber_extras.sh' >> nohup_cyber_qwen.out 2>&1
rc=$?
echo "[$(date)] link $LINK driver exit $rc" >> nohup_cyber_qwen.out

if grep -q "=== CYBER QWEN EXTRAS DONE" nohup_cyber_qwen.out; then
  echo "cyber qwen campaign complete" >> nohup_cyber_qwen.out
elif [ $rc -eq 124 ] && [ $LINK -lt $MAXLINK ]; then
  CY_LINK=$((LINK+1)) sbatch --export=ALL,CY_LINK=$((LINK+1)) run_cyber_qwen.sbatch >> nohup_cyber_qwen.out 2>&1
else
  echo "CHAIN STOPPED (rc=$rc, link=$LINK) - needs attention" >> nohup_cyber_qwen.out
fi
