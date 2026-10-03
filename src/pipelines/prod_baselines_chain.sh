#!/bin/bash
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
# wait for the CIL chain to finish first
while pgrep -f "prod_cil_chain.sh|prod_unlearn.py" > /dev/null; do sleep 300; done

collapsed () {  # tag -> 0 if all three split BLEUs ~ 0 (model destroyed)
  python3 - "$1" <<'PY'
import json, sys
r = json.load(open("prod_results.json")).get(sys.argv[1])
sys.exit(0 if r and all(v["bleu"] < 0.02 for v in r.values()) else 1)
PY
}

for m in prod ga gd dpo npo simnpo ila; do
  if [ ! -d adapters_prod/$m/epoch5 ]; then
    python3 -u prod_baselines.py --method $m || { echo "TRAIN $m FAILED"; continue; }
  fi
  for ep in 1 3 5; do
    tag=${m}_ep${ep}
    python3 -c "import json,sys;sys.exit(0 if '$tag' in json.load(open('prod_results.json')) else 1)" 2>/dev/null && continue
    python3 -u prod_eval.py --tag $tag --adapter adapters_prod/$m/epoch$ep \
      || echo "EVAL $tag FAILED"
    if collapsed "$tag"; then echo "=== $m collapsed at ep$ep, skipping rest ==="; break; fi
  done
done
echo "=== PROD BASELINES SUMMARY ==="
python3 - <<'PY'
import json
r = json.load(open("prod_results.json"))
for tag in sorted(r):
    v = r[tag]
    print(f"{tag:at}" if False else f"{tag:16}",
          " ".join(f"{s}:{v[s]['bleu']:.2f}" for s in ("forget","heldout","retain")))
PY
echo PROD_BASELINES_DONE
