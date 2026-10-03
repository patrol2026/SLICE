#!/bin/bash
# CodeLlama-7B LeetCode campaign (both splits, all methods incl. PROD, eval ep5).
#   funnel(generation) -> splits A/B -> n-gram annotate -> mutants -> train+eval
# Idempotent, GPU-guarded. Assumes generation writes results_leetcode_codellama.jsonl.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LC_MODEL=codellama/CodeLlama-7b-Instruct-hf
export LC_RESULTS=results_leetcode_codellama.jsonl
export LC_LORA_TARGETS=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
POOL=results_leetcode_codellama.jsonl
export LC_FORGET_ANN=ann_full_cl.jsonl
export LC_NGRAM_CORPUS=$POOL
export LC_RESULTS_DIR=results_codellama       # keep CodeLlama results separate
mkdir -p "$LC_RESULTS_DIR"

wait_gpu () { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 8000 ]; do echo "GPU busy ($(date '+%T'))"; sleep 120; done; }
have () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('results_codellama/lc_summary.json')) else 1)" 2>/dev/null; }

# 0. wait for the generation funnel to finish
while pgrep -f "run_leetcode.py" >/dev/null 2>&1; do echo "waiting for generation to finish ($(date '+%T'))"; sleep 120; done
NSOLVED=$(python3 -c "import json;print(sum(1 for l in open('$POOL') if json.loads(l).get('passed')))")
echo "=== CodeLlama solved pool: $NSOLVED ==="

# 1. splits A (10/20/70) and B (20/20/60), seed 42
python3 - <<PY
import json, random
def make(fr,hd,out):
    solved=sorted(json.loads(l)['task_id'] for l in open('$POOL') if json.loads(l).get('passed'))
    rng=random.Random(42); rng.shuffle(solved); n=len(solved)
    nf,nh=round(fr*n),round(hd*n)
    s={'forget':sorted(solved[:nf]),'heldout':sorted(solved[nf:nf+nh]),'retain':sorted(solved[nf+nh:])}
    json.dump(s,open(out,'w')); print(out,{k:len(v) for k,v in s.items()},'of',n)
make(0.10,0.20,'splits_cl_A.json')
make(0.20,0.20,'splits_cl_B.json')
PY

# 2. n-gram annotate the solved solutions (idf corpus = CodeLlama pool)
[ -f ann_full_cl.jsonl ] || python3 -u ngram_annotate.py $POOL ann_full_cl.jsonl generated_solution

# 3. verified mutants per split
[ -f lc_mutants_clA.json ] || LC_SPLITS=splits_cl_A.json LC_MUTANTS_OUT=lc_mutants_clA.json python3 -u gen_mutants_lc.py
[ -f lc_mutants_clB.json ] || LC_SPLITS=splits_cl_B.json LC_MUTANTS_OUT=lc_mutants_clB.json python3 -u gen_mutants_lc.py

# 4. train all methods x both splits, eval EP5 only
for cs in "clA splits_cl_A.json" "clB splits_cl_B.json"; do
  set -- $cs; cell=$1; splits=$2
  export LC_MUTANTS=lc_mutants_${cell}.json
  outdir=adapters_grid/${cell}
  for m in ga gd dpo npo simnpo codeeraser prod slice; do
    if [ ! -d $outdir/$m/epoch5 ]; then
      wait_gpu
      echo "=== TRAIN $m/$cell ($(date '+%F %T')) ==="
      python3 -u unlearn_lc.py --method $m --splits $splits --outdir $outdir --mkey _${cell} \
        || { echo "TRAIN $m/$cell FAILED"; continue; }
    else
      echo "=== train $m/$cell already done ==="
    fi
    tag=${m}_${cell}_ep5
    if have $tag; then echo "=== eval $tag already done ==="; continue; fi
    wait_gpu
    echo "=== EVAL $tag ($(date '+%F %T')) ==="
    python3 -u eval_lc.py --tag $tag --batch-size 10 --adapter $outdir/$m/epoch5 --splits $splits \
      || echo "EVAL $tag FAILED"
  done
done

echo "=== CODELLAMA GRID DONE ($(date '+%F %T')) ==="
python3 - <<'PY'
import json
s=json.load(open('results_codellama/lc_summary.json'))
for tag in sorted(s):
    if tag.endswith('_ep5') and ('_clA_' in tag or '_clB_' in tag):
        v=s[tag]; print(f"{tag:26}", " ".join(f"{k}:{d['pass@1']:.2f}" for k,d in v.items()))
PY
