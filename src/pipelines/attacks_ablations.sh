#!/bin/bash
# RQ3 attacks (MIA, paraphrase, relearning) + RQ4 loss-component ablations
# for all 4 cells. Waits politely for the GPU; idempotent; stats to the same
# logs as the grid.
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True

LOG=grid_stage_log.jsonl
stage () {
  local name="$1"; shift
  local t0=$(date +%s)
  echo "=== STAGE $name ($(date '+%F %T')) ==="
  "$@"; local rc=$?
  echo "{\"stage\":\"$name\",\"start\":$t0,\"seconds\":$(( $(date +%s) - t0 )),\"rc\":$rc}" >> $LOG
  return $rc
}
wait_gpu () {
  while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -ge 2500 ]; do
    sleep 180
  done
}
have_tag () { python3 -c "import json,sys;sys.exit(0 if '$1' in json.load(open('$2')) else 1)" 2>/dev/null; }

qwen_env () { unset LC_MODEL LC_RESULTS LC_LORA_TARGETS LC_GEN_BATCH; }
ds_env () {
  export LC_GEN_BATCH=32
  export LC_MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
  export LC_RESULTS=results_leetcode_dsv2.jsonl
  export LC_LORA_TARGETS=q_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj
}
cell_env () {  # cell -> splits file + model env + mutants/ann
  case $1 in
    qwenA) qwen_env; SPLITS=splits_qwen_A.json;    export LC_MUTANTS=lc_mutants_qwenA.json LC_FORGET_ANN=ann_full_qwen.jsonl;;
    qwenB) qwen_env; SPLITS=leetcode_splits.json;  export LC_MUTANTS=lc_mutants_qwenB.json LC_FORGET_ANN=ann_full_qwen.jsonl;;
    dsA)   ds_env;   SPLITS=leetcode_splits_ds.json; export LC_MUTANTS=lc_mutants_dsA.json LC_FORGET_ANN=ann_full_ds.jsonl;;
    dsB)   ds_env;   SPLITS=splits_ds_B.json;      export LC_MUTANTS=lc_mutants_dsB.json LC_FORGET_ANN=ann_full_ds.jsonl;;
  esac
  export LC_SPLITS=$SPLITS
}

for cell in qwenA qwenB dsA dsB; do
  cell_env $cell
  SLICE=adapters_grid/${cell}/slice/epoch5

  # ---------- RQ4: loss-component ablations ----------
  for ab in noil noforget noguard nonll; do
    outdir=adapters_ab/${cell}_${ab}
    if [ ! -d $outdir/slice/epoch5 ]; then
      wait_gpu
      stage "train_slice-${ab}_${cell}" python3 -u unlearn_lc.py --method slice \
        --ablation $ab --splits $SPLITS --outdir $outdir --mkey _${cell}_${ab} \
        || { echo "TRAIN ${ab}/${cell} FAILED"; continue; }
    fi
    for ep in 1 3 5; do
      tag=slice-${ab}_${cell}_ep${ep}
      have_tag $tag lc_summary.json && continue
      wait_gpu
      stage "eval_${tag}" python3 -u eval_lc.py --tag $tag --batch-size 32 \
        --adapter $outdir/slice/epoch$ep --splits $SPLITS \
        || echo "EVAL $tag FAILED"
    done
  done

  # ---------- RQ3a: MIA ----------
  have_tag "base_${cell}" mia_scores.json || { wait_gpu
    stage "mia_base_${cell}" python3 -u mia_lc.py --tag base_${cell}; }
  have_tag "slice_${cell}_ep5" mia_scores.json || { wait_gpu
    stage "mia_slice_${cell}" python3 -u mia_lc.py --tag slice_${cell}_ep5 --adapter $SLICE; }

  # ---------- RQ3b: paraphrase (needs per-cell paraphrase file) ----------
  PFILE=lc_paraphrases_${cell}.json
  if [ -f $PFILE ]; then
    export LC_PARAS=$PFILE
    for who in base slice; do
      tag=para_${who}_${cell}
      have_tag $tag paraphrase_results.json && continue
      wait_gpu
      if [ $who = base ]; then
        stage "$tag" python3 -u paraphrase_probe.py --tag $tag || echo "$tag FAILED"
      else
        stage "$tag" python3 -u paraphrase_probe.py --tag $tag --adapter $SLICE || echo "$tag FAILED"
      fi
    done
  else
    echo "=== $PFILE missing — paraphrase for $cell deferred ==="
  fi

  # ---------- RQ3c: relearning ----------
  have_tag "relearn_slice_${cell}" relearn_results.json || { wait_gpu
    stage "relearn_${cell}" python3 -u relearn_lc.py --tag relearn_slice_${cell} \
      --adapter $SLICE || echo "RELEARN $cell FAILED"; }
done

echo "=== MIA AUC ==="
python3 mia_lc.py --auc || true
echo "=== ATTACKS+ABLATIONS DONE ==="; touch ATTACKS_DONE
