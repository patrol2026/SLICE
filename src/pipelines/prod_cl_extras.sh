#!/bin/bash
# CodeLlama extras: utility + attacks for BOTH splits and ALL methods, final
# epoch (epoch5) only. Runs after run_prod_codellama.sh / prod_attacks_cl.sh.
# Idempotent: finished rows are skipped.
#
# Utility  -> utility_results.json
#   prodtask_cl_memorized            (shared by both splits)
#   prodtask_cl_<m>                  split B  (same naming as the driver rows)
#   prodtask_cl_A_<m>                split A
# Attacks  -> prod_attack_results_cl.json (B), prod_attack_results_cl_A.json (A)
#   mia_<m>, prefix_<m> for memorized + all methods; relearn_<m> for methods;
#   mia_auc recomputed at the end of each split.
cd "$(dirname "$0")"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PROD_MODEL="codellama/CodeLlama-7b-Instruct-hf"
export PROD_LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
export PROD_MEM="adapters_prod_cl/memorized/epoch10"
export PROD_MEMCENSUS="prod_memorization_cl.jsonl"
export LC_MODEL="$PROD_MODEL" LC_GEN_BATCH=16
MEM=adapters_prod_cl/memorized/epoch10
METHODS="slice prod ga gd dpo npo simnpo codeeraser"
EP=epoch5

have () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('$2')) else 1)" 2>/dev/null; }
fail=0

# ---------------- utility (all methods, both splits) ----------------
echo "=== EXTRAS: UTILITY ==="
have prodtask_cl_memorized utility_results.json || python3 -u util_suite.py --tag prodtask_cl_memorized --adapter $MEM || fail=1
for T in B A; do
  for m in $METHODS; do
    if [ $T = B ]; then tag=prodtask_cl_$m; else tag=prodtask_cl_A_$m; fi
    have $tag utility_results.json || python3 -u util_suite.py --tag $tag --adapter $MEM,adapters_prod_cl_$T/$m/$EP || fail=1
  done
done

# ---------------- attacks (all methods, both splits) ----------------
for T in B A; do
  echo "=== EXTRAS: ATTACKS split $T ==="
  export PROD_SPLITS=prod_splits_cl_$T.json
  if [ $T = B ]; then export PROD_ATTACK_RESULTS=prod_attack_results_cl.json
  else export PROD_ATTACK_RESULTS=prod_attack_results_cl_A.json; fi
  R=$PROD_ATTACK_RESULTS
  have mia_memorized $R    || python3 -u prod_attacks.py mia --tag memorized --adapters $MEM || fail=1
  have prefix_memorized $R || python3 -u prod_attacks.py prefix --tag memorized --adapters $MEM || fail=1
  for m in $METHODS; do
    A=$MEM,adapters_prod_cl_$T/$m/$EP
    have mia_$m $R    || python3 -u prod_attacks.py mia --tag $m --adapters $A || fail=1
    have prefix_$m $R || python3 -u prod_attacks.py prefix --tag $m --adapters $A || fail=1
  done
  python3 -u prod_attacks.py auc || fail=1          # recompute over all mia_* rows
  for m in $METHODS; do
    have relearn_$m $R || python3 -u prod_attacks.py relearn --tag $m --adapters $MEM,adapters_prod_cl_$T/$m/$EP || fail=1
  done
done

[ $fail = 0 ] || { echo "EXTRAS INCOMPLETE (a step failed)"; exit 1; }
echo "=== PROD CODELLAMA EXTRAS COMPLETE ==="
