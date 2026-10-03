#!/bin/bash
# Unlearning audits on the s10 live operating points:
#   models: base, GD ep2, CodeEraser ep2, CIL ep5
#   1) MIA scores + AUC   2) paraphrase probe   3) relearning attack (GD, CIL)
#   4) utility suite (HumanEval, MBPP, MMLU-500, GSM8K-200)
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True

GD=adapters_lc_s10/gd/epoch2
CodeEraser=adapters_lc_s10/ila/epoch2
CIL=adapters_lc_s10/cil/epoch5

echo "=== MIA scores ==="
python mia_lc.py --tag base || exit 1
python mia_lc.py --tag gd_s10_ep2 --adapter $GD
python mia_lc.py --tag ila_s10_ep2 --adapter $CodeEraser
python mia_lc.py --tag cil_s10_ep5 --adapter $CIL
echo "=== MIA AUC ==="
python mia_lc.py --auc

echo "=== merge paraphrases ==="
python - <<'PY'
import json, glob
files = sorted(glob.glob('./scratch/lc_para/para_*.json'))
assert len(files) == 5, f"only {len(files)} paraphrase files ready"
paras = {}
for f in files:
    paras.update(json.load(open(f)))
json.dump(paras, open('lc_paraphrases.json', 'w'))
print(f"{len(paras)} paraphrases merged")
PY

echo "=== PARAPHRASE probe ==="
python paraphrase_probe.py --tag base
python paraphrase_probe.py --tag gd_s10_ep2 --adapter $GD
python paraphrase_probe.py --tag ila_s10_ep2 --adapter $CodeEraser
python paraphrase_probe.py --tag cil_s10_ep5 --adapter $CIL

echo "=== RELEARN cil ==="
python relearn_lc.py --tag cil_s10_ep5 --adapter $CIL
echo "=== RELEARN gd ==="
python relearn_lc.py --tag gd_s10_ep2 --adapter $GD

echo "=== UTILITY suite ==="
python util_suite.py --tag base
python util_suite.py --tag gd_s10_ep2 --adapter $GD
python util_suite.py --tag ila_s10_ep2 --adapter $CodeEraser
python util_suite.py --tag cil_s10_ep5 --adapter $CIL

echo "=== AUDIT DONE ==="
cat mia_auc.json paraphrase_results.json relearn_results.json utility_results.json | head -100
