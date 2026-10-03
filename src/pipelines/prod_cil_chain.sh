#!/bin/bash
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
have () { python3 -c "import json;assert '$1' in json.load(open('prod_results.json'))" 2>/dev/null; }
have memorized || python3 -u prod_eval.py --tag memorized
if [ ! -d adapters_prod/cil/epoch5 ]; then
  python3 -u prod_unlearn.py || { echo TRAIN_FAILED; exit 1; }
fi
for ep in 1 3 5; do
  have cil_ep${ep} || python3 -u prod_eval.py --tag cil_ep${ep} --adapter adapters_prod/cil/epoch${ep}
done
echo PROD_CIL_CHAIN_DONE
