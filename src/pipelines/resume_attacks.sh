#!/bin/bash
cd /path/to/slice
# wait for the orphaned eval to finish, then resume the idempotent pipeline
while pgrep -f "eval_lc.py --tag cil-noforget_dsB_ep3" > /dev/null; do sleep 120; done
bash attacks_ablations.sh
