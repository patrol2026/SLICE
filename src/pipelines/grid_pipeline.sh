#!/bin/bash
# Master pipeline: detector study v2 -> splits -> mutants -> full unlearning
# grid (7 methods x {qwen,ds} x {A=10/20/70, B=20/20/60}), per-epoch evals with
# 0/0/0 early stop. All statistics land in lc_metrics.json / lc_summary.json /
# iline_stats_v2.json plus stage timing in grid_stage_log.jsonl.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True

LOG=grid_stage_log.jsonl
stage () {  # stage <name> <cmd...>
  local name="$1"; shift
  local t0=$(date +%s)
  echo "=== STAGE $name ($(date '+%F %T')) ==="
  "$@"; local rc=$?
  local t1=$(date +%s)
  echo "{\"stage\":\"$name\",\"start\":$t0,\"seconds\":$((t1-t0)),\"rc\":$rc}" >> $LOG
  return $rc
}

all_zero () {
  python3 - "$1" <<'PY'
import json, sys
s = json.load(open("lc_summary.json")).get(sys.argv[1])
sys.exit(0 if s and sum(v["passed"] for v in s.values()) == 0 else 1)
PY
}

wait_gpu () {  # block until the GPU has room (other experiments may be running)
  local used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 8000 ] && break
    echo "GPU busy (${used} MiB) — waiting 2 min ($(date '+%T'))"
    sleep 120
  done
}

train_eval () {  # train_eval <method> <cell> <splits> <extra-train-args...>
  local m=$1 cell=$2 splits=$3; shift 3
  local outdir=adapters_grid/${cell}
  if [ ! -d $outdir/$m/epoch5 ]; then
    wait_gpu
    stage "train_${m}_${cell}" python3 -u unlearn_lc.py --method $m \
      --splits $splits --outdir $outdir --mkey _${cell} "$@" \
      || { echo "TRAIN $m/$cell FAILED"; return 1; }
  else
    echo "=== train $m/$cell already done ==="
  fi
  for ep in 1 2 3 4 5; do
    local tag=${m}_${cell}_ep${ep}
    if python3 -c "import json,sys;sys.exit(0 if '$tag' in json.load(open('lc_summary.json')) else 1)" 2>/dev/null; then
      echo "=== eval $tag already done ==="
    else
      wait_gpu
      stage "eval_${tag}" python3 -u eval_lc.py --tag $tag --batch-size 10 \
        --adapter $outdir/$m/epoch$ep --splits $splits \
        || { echo "EVAL $tag FAILED"; continue; }
    fi
    if all_zero "$tag"; then
      echo "=== $tag is 0/0/0 — skipping remaining epochs ==="; break
    fi
  done
}

# ---------- 1. detector study v2 (RQ1) on full validated corpus ----------
if [ ! -f iline_stats_v2.json ]; then
  stage detectors_v2 env LC_DETECT_REF=ann_full_qwen.jsonl LC_DETECT_OUT=_v2 \
    python3 -u run_detectors.py || echo "DETECTORS FAILED (continuing)"
fi

# ---------- 2. splits ----------
[ -f splits_qwen_A.json ] || stage make_splits python3 make_splits_v2.py

# ---------- 3. mutants (validated annotations) per cell ----------
mut () {  # mut <cell> <splits> <ann> <out>
  [ -f $4 ] && { echo "=== mutants $1 exist ==="; return; }
  stage "mutants_$1" env LC_SPLITS=$2 LC_FORGET_ANN=$3 LC_MUTANTS_OUT=$4 \
    python3 -u gen_mutants_lc.py
}
mut qwenA splits_qwen_A.json      ann_full_qwen.jsonl lc_mutants_qwenA.json
mut qwenB leetcode_splits.json    ann_full_qwen.jsonl lc_mutants_qwenB.json
mut dsA   leetcode_splits_ds.json ann_full_ds.jsonl   lc_mutants_dsA.json
mut dsB   splits_ds_B.json        ann_full_ds.jsonl   lc_mutants_dsB.json

# ---------- 4. unlearning grid ----------
# QWEN cells
export LC_FORGET_ANN=ann_full_qwen.jsonl
for cell_splits in "qwenA splits_qwen_A.json" "qwenB leetcode_splits.json"; do
  set -- $cell_splits; cell=$1; splits=$2
  export LC_MUTANTS=lc_mutants_${cell}.json
  for m in ga gd dpo npo simnpo codeeraser slice; do
    # qwenB baselines ga..npo already exist under old tags with same split —
    # rerun only annotation-dependent + new methods there
    if [ $cell = qwenB ] && [[ $m =~ ^(ga|gd|dpo|npo)$ ]]; then
      echo "=== $m/qwenB covered by legacy tags (${m}_ep*) — skipping ==="
      continue
    fi
    train_eval $m $cell $splits
  done
done

# DS cells
export LC_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
export LC_RESULTS=results_leetcode_dsv2.jsonl
export LC_FORGET_ANN=ann_full_ds.jsonl
export LC_LORA_TARGETS=q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj
for cell_splits in "dsA leetcode_splits_ds.json" "dsB splits_ds_B.json"; do
  set -- $cell_splits; cell=$1; splits=$2
  export LC_MUTANTS=lc_mutants_${cell}.json
  for m in ga gd dpo npo simnpo codeeraser slice; do
    if [ $cell = dsA ] && [[ $m =~ ^(ga|gd|dpo|npo)$ ]]; then
      echo "=== $m/dsA covered by legacy tags (${m}_ds_ep*) — skipping ==="
      continue
    fi
    train_eval $m $cell $splits
  done
done

echo "=== GRID SUMMARY (forget/heldout/retain pass@1) ==="
python3 - <<'PY'
import json
s = json.load(open('lc_summary.json'))
for tag in sorted(s):
    if any(c in tag for c in ('qwenA','qwenB','dsA','dsB')):
        v = s[tag]
        print(f"{tag:26} " + " ".join(f"{k}:{d['pass@1']:.2f}" for k, d in v.items()))
PY
echo "=== ALL DONE ==="
