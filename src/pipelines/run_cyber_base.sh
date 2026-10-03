#!/bin/bash
# Fill the base-model references for the CyberSecEval CodeLlama task:
# base utility (1 run, split-independent) + base attacks (MIA, prefix) per split.
# Writes to results_cyber_cl/. Idempotent.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CY_MODEL=codellama/CodeLlama-7b-Instruct-hf
export CY_RESULTS_DIR=results_cyber_cl
export LC_MODEL=codellama/CodeLlama-7b-Instruct-hf
export LC_RESULTS_DIR=results_cyber_cl
export LC_GEN_BATCH=16
mkdir -p "$CY_RESULTS_DIR"

wait_gpu () { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 8000 ]; do sleep 120; done; }
uhave () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('$CY_RESULTS_DIR/utility_results.json')) else 1)" 2>/dev/null; }
ahave () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('$CY_RESULTS_DIR/cyber_attacks_$2.json')) else 1)" 2>/dev/null; }

# base utility (raw CodeLlama, no adapter) - reference for both splits
uhave cyber_base || { wait_gpu; python3 -u util_suite.py --tag cyber_base || echo "UTIL base FAILED"; }

# base attacks per split (adapters="" -> raw base model)
for s in A B; do
  ahave mia_base $s    || { wait_gpu; python3 -u cyber_attacks.py mia --split $s --tag base --adapters "" || echo "MIA base/$s FAILED"; }
  ahave prefix_base $s || { wait_gpu; python3 -u cyber_attacks.py prefix --split $s --tag base --adapters "" || echo "PREFIX base/$s FAILED"; }
  # recompute AUC so base is included as a control vs the memorized reference
  wait_gpu; python3 -u cyber_attacks.py mia_auc --split $s || echo "MIA_AUC/$s FAILED"
done

echo "=== CYBER BASE REFERENCES DONE ($(date '+%F %T')) ==="
python3 -c "
import json
print('utility rows:', list(json.load(open('results_cyber_cl/utility_results.json'))))
for s in ('A','B'):
    print(f'attacks {s}:', list(json.load(open(f'results_cyber_cl/cyber_attacks_{s}.json'))))
"
