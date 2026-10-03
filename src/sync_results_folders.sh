#!/bin/bash
# Collect each model's campaign outputs into its own folder, so DeepSeek and
# CodeLlama results are kept apart (the live pipeline keeps writing to the
# flat filenames in the project root; this only copies).
#
#   bash sync_results_folders.sh            # sync all models
#   bash sync_results_folders.sh ds         # just deepseek (ds | cl | qwen)
#
# Per-model folder gets: results/splits/census/attacks + that model's rows
# extracted from the shared utility_results.json and prod_metrics.json.
# Re-run any time; it overwrites only inside results_<model>/.
cd "$(dirname "$0")"

sync_one () {   # $1=suffix used in filenames (ds|cl|"")  $2=folder  $3=metrics/tag key
  local S=$1 DIR=$2 KEY=$3
  mkdir -p "$DIR"
  if [ -n "$S" ]; then
    cp -f prod_results_${S}_B.json prod_results_${S}_A.json "$DIR"/ 2>/dev/null
    cp -f prod_memorization_${S}.jsonl prod_funnel_summary_${S}.json "$DIR"/ 2>/dev/null
    cp -f prod_splits_${S}_B.json prod_splits_${S}_A.json "$DIR"/ 2>/dev/null
    cp -f prod_cil_data_${S}_B.json prod_cil_data_${S}_A.json "$DIR"/ 2>/dev/null
    cp -f prod_attack_results_${S}.json prod_attack_results_${S}_A.json "$DIR"/ 2>/dev/null
    cp -f nohup_prod_${S}.out "$DIR"/run_log.txt 2>/dev/null
  else   # qwen = the original unsuffixed files
    cp -f prod_results.json prod_results_A.json prod_memorization.jsonl \
          prod_funnel_summary.json prod_splits.json prod_splits_A.json \
          prod_cil_data.json prod_cil_data_A.json prod_attack_results.json "$DIR"/ 2>/dev/null
    cp -f nohup_prod_chain.out "$DIR"/run_log.txt 2>/dev/null
  fi
  KEY="$KEY" DIR="$DIR" python3 - <<'PY'
import json, os
key, d = os.environ["KEY"], os.environ["DIR"]
def dump(src, out, pred):
    if not os.path.exists(src): return
    r = json.load(open(src))
    sub = {k: v for k, v in r.items() if pred(k)}
    json.dump(sub, open(os.path.join(d, out), "w"), indent=1)
    print(f"  {out}: {len(sub)} rows")
dump("utility_results.json", "utility_results.json", lambda k: k.startswith(f"prodtask_{key}") if key else (k.startswith("prodtask_") and "_ds" not in k and "_cl" not in k))
dump("prod_metrics.json", "prod_metrics.json", lambda k: (f"_{key}_" in k or f"_{key}." in k or k.endswith(f"_{key}") or f"prodtask_{key}" in k) if key else ("_ds" not in k and "_cl" not in k))
PY
  echo "-> $DIR"
}

case "${1:-all}" in
  ds)   sync_one ds results_deepseek ds ;;
  cl)   sync_one cl results_codellama cl ;;
  qwen) sync_one "" results_qwen "" ;;
  all)  sync_one ds results_deepseek ds; sync_one cl results_codellama cl; sync_one "" results_qwen "" ;;
  *)    echo "usage: $0 [ds|cl|qwen|all]"; exit 1 ;;
esac
