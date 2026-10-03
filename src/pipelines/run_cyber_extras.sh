#!/bin/bash
# CyberSecEval CodeLlama EXTRAS: utility + attacks (MIA, prefix, relearn) for
# ALL methods, both splits. Waits for the main run. Writes to results_cyber_cl/.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CY_MODEL=codellama/CodeLlama-7b-Instruct-hf
export CY_MEM=adapters_cyber_cl/memorized/epoch10
export CY_RESULTS_DIR=results_cyber_cl
export LC_MODEL=codellama/CodeLlama-7b-Instruct-hf
export LC_RESULTS_DIR=results_cyber_cl
export LC_GEN_BATCH=16
MEM=adapters_cyber_cl/memorized/epoch10
ADIRBASE=adapters_cyber           # split adapters: ${ADIRBASE}_${s}
METHODS="ga gd dpo npo simnpo ila prod cil"
mkdir -p "$CY_RESULTS_DIR"

wait_gpu () { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 8000 ]; do sleep 120; done; }
uhave () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('$CY_RESULTS_DIR/utility_results.json')) else 1)" 2>/dev/null; }
ahave () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('$CY_RESULTS_DIR/cyber_attacks_$2.json')) else 1)" 2>/dev/null; }

while ! grep -q "CYBER CODELLAMA DONE" nohup_cyber_grid.out 2>/dev/null; do
  echo "extras: waiting for main run ($(date '+%T'))"; sleep 180; done

# ---------- utility ----------
uhave cyber_base || { wait_gpu; python3 -u util_suite.py --tag cyber_base || echo "UTIL base FAILED"; }
uhave cyber_memorized || { wait_gpu; python3 -u util_suite.py --tag cyber_memorized --adapter $MEM || echo "UTIL mem FAILED"; }
for s in A B; do
  for m in $METHODS; do
    A=${ADIRBASE}_${s}/${m}/epoch5
    [ -d $A ] || { echo "no adapter $A, skip util"; continue; }
    uhave cyber_${m}_${s} || { wait_gpu; python3 -u util_suite.py --tag cyber_${m}_${s} --adapter $MEM,$A || echo "UTIL ${m}_$s FAILED"; }
  done
done

# ---------- attacks per split, all methods ----------
for s in A B; do
  ahave mia_memorized $s || { wait_gpu; python3 -u cyber_attacks.py mia --split $s --tag memorized --adapters $MEM || echo "MIA mem/$s FAILED"; }
  for m in $METHODS; do
    A=${ADIRBASE}_${s}/${m}/epoch5; [ -d $A ] || continue
    ahave mia_${m} $s || { wait_gpu; python3 -u cyber_attacks.py mia --split $s --tag ${m} --adapters $MEM,$A || echo "MIA ${m}/$s FAILED"; }
  done
  wait_gpu; python3 -u cyber_attacks.py mia_auc --split $s || echo "MIA_AUC/$s FAILED"

  ahave prefix_memorized $s || { wait_gpu; python3 -u cyber_attacks.py prefix --split $s --tag memorized --adapters $MEM || echo "PREFIX mem/$s FAILED"; }
  for m in $METHODS; do
    A=${ADIRBASE}_${s}/${m}/epoch5; [ -d $A ] || continue
    ahave prefix_${m} $s || { wait_gpu; python3 -u cyber_attacks.py prefix --split $s --tag ${m} --adapters $MEM,$A || echo "PREFIX ${m}/$s FAILED"; }
  done

  for m in $METHODS; do
    A=${ADIRBASE}_${s}/${m}/epoch5; [ -d $A ] || continue
    ahave relearn_${m} $s || { wait_gpu; python3 -u cyber_attacks.py relearn --split $s --tag ${m} --adapters $MEM,$A || echo "RELEARN ${m}/$s FAILED"; }
  done
done

echo "=== CYBER CODELLAMA EXTRAS DONE ($(date '+%F %T')) ==="
