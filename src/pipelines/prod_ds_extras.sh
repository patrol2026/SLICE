#!/bin/bash
# DeepSeek extras: utility + attacks for BOTH splits and ALL methods, final
# epoch (epoch5) only. Runs after run_prod_deepseek.sh / prod_attacks_ds.sh.
# Idempotent: finished rows are skipped.
#
# Utility  -> utility_results.json
#   prodtask_ds_memorized            (shared by both splits)
#   prodtask_ds_<m>                  split B  (same naming as the driver rows)
#   prodtask_ds_A_<m>                split A
# Attacks  -> prod_attack_results_ds.json (B), prod_attack_results_ds_A.json (A)
#   mia_<m>, prefix_<m> for memorized + all methods; relearn_<m> for methods;
#   mia_auc recomputed at the end of each split.
cd "$(dirname "$0")"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PROD_MODEL="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
export PROD_LORA_TARGETS="q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj"
export PROD_MEM="adapters_prod_ds/memorized/epoch10"
export PROD_MEMCENSUS="prod_memorization_ds.jsonl"
export LC_MODEL="$PROD_MODEL" LC_GEN_BATCH=16
MEM=adapters_prod_ds/memorized/epoch10
METHODS="cil prod ga gd dpo npo simnpo ila"
EP=epoch5

have () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('$2')) else 1)" 2>/dev/null; }
fail=0

# ---------------- utility (all methods, both splits) ----------------
echo "=== EXTRAS: UTILITY ==="
have prodtask_ds_memorized utility_results.json || python3 -u util_suite.py --tag prodtask_ds_memorized --adapter $MEM || fail=1
for T in B A; do
  for m in $METHODS; do
    if [ $T = B ]; then tag=prodtask_ds_$m; else tag=prodtask_ds_A_$m; fi
    have $tag utility_results.json || python3 -u util_suite.py --tag $tag --adapter $MEM,adapters_prod_ds_$T/$m/$EP || fail=1
  done
done

# ---------------- attacks (all methods, both splits) ----------------
for T in B A; do
  echo "=== EXTRAS: ATTACKS split $T ==="
  export PROD_SPLITS=prod_splits_ds_$T.json
  if [ $T = B ]; then export PROD_ATTACK_RESULTS=prod_attack_results_ds.json
  else export PROD_ATTACK_RESULTS=prod_attack_results_ds_A.json; fi
  R=$PROD_ATTACK_RESULTS
  have mia_memorized $R    || python3 -u prod_attacks.py mia --tag memorized --adapters $MEM || fail=1
  have prefix_memorized $R || python3 -u prod_attacks.py prefix --tag memorized --adapters $MEM || fail=1
  for m in $METHODS; do
    A=$MEM,adapters_prod_ds_$T/$m/$EP
    have mia_$m $R    || python3 -u prod_attacks.py mia --tag $m --adapters $A || fail=1
    have prefix_$m $R || python3 -u prod_attacks.py prefix --tag $m --adapters $A || fail=1
  done
  python3 -u prod_attacks.py auc || fail=1          # recompute over all mia_* rows
  for m in $METHODS; do
    have relearn_$m $R || python3 -u prod_attacks.py relearn --tag $m --adapters $MEM,adapters_prod_ds_$T/$m/$EP || fail=1
  done
done

[ $fail = 0 ] || { echo "EXTRAS INCOMPLETE (a step failed)"; exit 1; }
echo "=== PROD DEEPSEEK EXTRAS COMPLETE ==="
