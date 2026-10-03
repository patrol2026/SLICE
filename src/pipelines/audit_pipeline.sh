#!/bin/bash
# Unlearning audits on the s10 live operating points:
#   models: base, GD ep2, CodeEraser ep2, SLICE ep5
#   1) MIA scores + AUC   2) paraphrase probe   3) relearning attack (GD, SLICE)
#   4) utility suite (HumanEval, MBPP, MMLU-500, GSM8K-200)
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True

GD=adapters_lc_s10/gd/epoch2
CodeEraser=adapters_lc_s10/codeeraser/epoch2
SLICE=adapters_lc_s10/slice/epoch5

echo "=== MIA scores ==="
python mia_lc.py --tag base || exit 1
python mia_lc.py --tag gd_s10_ep2 --adapter $GD
python mia_lc.py --tag codeeraser_s10_ep2 --adapter $CodeEraser
python mia_lc.py --tag slice_s10_ep5 --adapter $SLICE
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
python paraphrase_probe.py --tag codeeraser_s10_ep2 --adapter $CodeEraser
python paraphrase_probe.py --tag slice_s10_ep5 --adapter $SLICE

echo "=== RELEARN slice ==="
python relearn_lc.py --tag slice_s10_ep5 --adapter $SLICE
echo "=== RELEARN gd ==="
python relearn_lc.py --tag gd_s10_ep2 --adapter $GD

echo "=== UTILITY suite ==="
python util_suite.py --tag base
python util_suite.py --tag gd_s10_ep2 --adapter $GD
python util_suite.py --tag codeeraser_s10_ep2 --adapter $CodeEraser
python util_suite.py --tag slice_s10_ep5 --adapter $SLICE

echo "=== AUDIT DONE ==="
cat mia_auc.json paraphrase_results.json relearn_results.json utility_results.json | head -100
