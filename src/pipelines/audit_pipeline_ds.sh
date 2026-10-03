#!/bin/bash
# DeepSeek-Coder-V2-Lite audits on CIL ep5 (+ GD ep2 for comparison).
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LC_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
export LC_RESULTS=results_leetcode_dsv2.jsonl
export LC_SPLITS=leetcode_splits_ds.json
export LC_GEN_BATCH=6            # 16B model — smaller generation batches
CIL=adapters_lc_ds/cil/epoch5
GD=adapters_lc_ds/gd/epoch4

echo "=== MIA scores (ds) ==="
python mia_lc.py --tag base_ds || echo "MIA base FAILED"
python mia_lc.py --tag cil_ds_ep5 --adapter $CIL || echo "MIA cil FAILED"
python mia_lc.py --tag gd_ds_ep4 --adapter $GD || echo "MIA gd FAILED"
echo "=== MIA AUC (ds) ==="
python mia_lc.py --auc || echo "MIA AUC FAILED"

echo "=== RELEARN cil (ds) ==="
python relearn_lc.py --tag cil_ds_ep5 --adapter $CIL || echo "RELEARN cil FAILED"

echo "=== UTILITY (ds) ==="
python util_suite.py --tag base_ds || echo "UTIL base FAILED"
python util_suite.py --tag cil_ds_ep5 --adapter $CIL || echo "UTIL cil FAILED"

# paraphrase: wait until all 4 DS paraphrase chunks are written, merge, then run
echo "=== waiting for DS paraphrases ==="
PARA_DIR=./scratch/ds_para
until [ "$(ls $PARA_DIR/para_*.json 2>/dev/null | wc -l)" -ge 4 ]; do sleep 30; done
python - <<'PY'
import json, glob
allp = dict(json.load(open('lc_paraphrases.json')))
for f in sorted(glob.glob('./scratch/ds_para/para_*.json')):
    allp.update(json.load(open(f)))
ds_forget = set(json.load(open('leetcode_splits_ds.json'))['forget'])
paras = {t: allp[t] for t in ds_forget if t in allp}   # ONLY DS forget
json.dump(paras, open('lc_paraphrases_ds.json','w'))
covered = len(paras)
print(f'ds paraphrases: {covered}/{len(ds_forget)} forget covered')
PY
export LC_PARAS=lc_paraphrases_ds.json
echo "=== PARAPHRASE probe (ds) ==="
python paraphrase_probe.py --tag base_ds || echo "PARA base FAILED"
python paraphrase_probe.py --tag cil_ds_ep5 --adapter $CIL || echo "PARA cil FAILED"

echo "=== DS AUDITS DONE ==="
cat mia_auc.json; echo; cat paraphrase_results.json | python -c "import json,sys; d=json.load(sys.stdin); print({k:v['paraphrase_pass@1'] for k,v in d.items() if '_ds' in k})"
