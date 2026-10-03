#!/bin/bash
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
MEM=adapters_prod/memorized/epoch10
CIL=$MEM,adapters_prod/cil/epoch5
PRODM=$MEM,adapters_prod/prod/epoch3

have_u () { python3 -c "import json;assert '$1' in json.load(open('utility_results.json'))" 2>/dev/null; }
have_a () { python3 -c "import json;assert '$1' in json.load(open('prod_attack_results.json'))" 2>/dev/null; }
have_r () { python3 -c "import json;assert '$1' in json.load(open('prod_results.json'))" 2>/dev/null; }

# ---- 1. utility suite (HumanEval/MBPP/MMLU/GSM8K) ----
have_u prodtask_memorized || python3 -u util_suite.py --tag prodtask_memorized --adapter $MEM
have_u prodtask_cil       || python3 -u util_suite.py --tag prodtask_cil --adapter $CIL
have_u prodtask_prod      || python3 -u util_suite.py --tag prodtask_prod --adapter $PRODM

# ---- 2. ablations (protocol B) ----
for ab in noil noforget noguard nonll; do
  if [ ! -d adapters_prod/cil_${ab}/epoch5 ]; then
    PROD_CIL_OUT=adapters_prod/cil_${ab} python3 -u prod_unlearn.py --ablation $ab \
      || { echo "ABL $ab FAILED"; continue; }
  fi
  for ep in 1 3 5; do
    have_r cil-${ab}_ep${ep} || python3 -u prod_eval.py --tag cil-${ab}_ep${ep} \
      --adapter adapters_prod/cil_${ab}/epoch${ep}
  done
done

# ---- 3. attacks ----
have_a mia_memorized || python3 -u prod_attacks.py mia --tag memorized --adapters $MEM
have_a mia_cil       || python3 -u prod_attacks.py mia --tag cil --adapters $CIL
have_a mia_prod      || python3 -u prod_attacks.py mia --tag prod --adapters $PRODM
python3 -u prod_attacks.py auc
have_a prefix_memorized || python3 -u prod_attacks.py prefix --tag memorized --adapters $MEM
have_a prefix_cil       || python3 -u prod_attacks.py prefix --tag cil --adapters $CIL
have_a prefix_prod      || python3 -u prod_attacks.py prefix --tag prod --adapters $PRODM
have_a relearn_cil      || python3 -u prod_attacks.py relearn --tag cil --adapters $CIL

echo PROD_EXTRAS_DONE
