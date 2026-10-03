#!/bin/bash
# Attacks (MIA / prefix / relearn) on the CodeLlama copyrighted-code models,
# mirroring the Qwen commands in prod_extras_chain.sh (protocol B adapters).
# Idempotent. Run after run_prod_codellama.sh.  -> prod_attack_results_cl.json
cd "$(dirname "$0")"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PROD_MODEL="codellama/CodeLlama-7b-Instruct-hf"
export PROD_LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
export PROD_SPLITS=prod_splits_cl_B.json
export PROD_ATTACK_RESULTS=prod_attack_results_cl.json

MEM=adapters_prod_cl/memorized/epoch10
SLICE=$MEM,adapters_prod_cl_B/slice/epoch5
PRODM=$MEM,adapters_prod_cl_B/prod/epoch5
have_a () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('$PROD_ATTACK_RESULTS')) else 1)" 2>/dev/null; }

echo "=== ATTACKS (codellama, split B) ==="
have_a mia_memorized || python3 -u prod_attacks.py mia --tag memorized --adapters $MEM
have_a mia_slice       || python3 -u prod_attacks.py mia --tag slice --adapters $SLICE
have_a mia_prod      || python3 -u prod_attacks.py mia --tag prod --adapters $PRODM
have_a mia_auc       || python3 -u prod_attacks.py auc
have_a prefix_memorized || python3 -u prod_attacks.py prefix --tag memorized --adapters $MEM
have_a prefix_slice       || python3 -u prod_attacks.py prefix --tag slice --adapters $SLICE
have_a prefix_prod      || python3 -u prod_attacks.py prefix --tag prod --adapters $PRODM
have_a relearn_slice      || python3 -u prod_attacks.py relearn --tag slice --adapters $SLICE

for k in mia_memorized mia_slice mia_prod mia_auc prefix_memorized prefix_slice prefix_prod relearn_slice; do
  have_a $k || { echo "ATTACKS INCOMPLETE (missing $k)"; exit 1; }
done
echo "=== PROD CODELLAMA ATTACKS COMPLETE ==="
