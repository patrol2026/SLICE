#!/bin/bash
# Full cross-lingual unlearning pipeline (HumanEval-X, Python/Java/C++).
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== MUTANTS (ast) ==="; python3 -u hex_mutants.py ast || exit 1
echo "=== MUTANTS (dataflow) ==="; python3 -u hex_mutants.py dataflow || exit 1

echo "=== BASELINE eval (no unlearning) ==="
python3 -u hex_eval.py --tag baseline || echo "BASELINE EVAL FAILED"

for DET in ast dataflow; do
  for MODE in mono multi; do
    echo "=== TRAIN $DET/$MODE ==="
    python3 -u hex_unlearn.py --detector $DET --mode $MODE --epochs 5 \
      || { echo "TRAIN $DET/$MODE FAILED"; continue; }
    echo "=== EVAL $DET/$MODE ==="
    python3 -u hex_eval.py --tag ${DET}_${MODE} --adapter adapters_hex/${DET}_${MODE} \
      || echo "EVAL $DET/$MODE FAILED"
  done
done

echo "=== CROSS-LINGUAL COMPARISON (forget pass@1 %, py / java / cpp) ==="
python3 - <<'PY'
import json
s=json.load(open('hex_summary.json'))
def row(tag):
    v=s.get(tag,{}).get('forget',{})
    def p(x): return f"{x*100:.0f}" if isinstance(x,(int,float)) else "--"
    return f"{p(v.get('python'))} / {p(v.get('java'))} / {p(v.get('cpp'))}"
print(f"{'condition':28} forget py/java/cpp")
print(f"{'baseline (no unlearn)':28} {row('baseline')}")
for det in ['ast','dataflow']:
    print(f"{det+' · python-only':28} {row(det+'_mono')}")
    print(f"{det+' · multilingual':28} {row(det+'_multi')}")
PY
echo "=== DONE ==="
