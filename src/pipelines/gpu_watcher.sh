#!/bin/bash
# Wait until the GPU is free (other experiments done), then launch the grid.
cd /path/to/slice
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  mutants_ready=$([ -f lc_mutants_qwenA.json ] && echo 1 || echo 0)
  if [ "$used" -lt 5000 ] && [ "$mutants_ready" = 1 ]; then
    echo "GPU free (${used} MiB) and mutants ready — launching grid at $(date)"
    setsid bash grid_pipeline.sh >> nohup_grid.out 2>&1 < /dev/null
    break
  fi
  sleep 120
done
