#!/bin/bash
cd /path/to/slice
grep -q PROD_EXTRAS_DONE nohup_prod_extras.out 2>/dev/null && exit 0
grep -q PROD_ABL_A_DONE nohup_prod_abl_A.out 2>/dev/null && exit 0
pgrep -f "prod_ablations_A_chain.sh" > /dev/null && exit 0
grep -q PROD_EXTRAS_DONE nohup_prod_extras.out 2>/dev/null && {
  grep -q PROD_ABL_A_DONE nohup_prod_abl_A.out 2>/dev/null || {
    echo "=== prod_keeper: relaunch A-ablations $(date '+%F %T') ===" >> nohup_prod_abl_A.out
    setsid bash prod_ablations_A_chain.sh >> nohup_prod_abl_A.out 2>&1 < /dev/null &
    exit 0
  }
}

pgrep -f "prod_extras_chain.sh|prod_attacks.py|util_suite.py" > /dev/null && exit 0
grep -q PROD_A_DONE nohup_prod_A.out 2>/dev/null && {
  echo "=== prod_keeper: relaunch extras $(date '+%F %T') ===" >> nohup_prod_extras.out
  setsid bash prod_extras_chain.sh >> nohup_prod_extras.out 2>&1 < /dev/null &
  exit 0
}
pgrep -f "prod_chain_A.sh" > /dev/null || {
  grep -q PROD_BASELINES_DONE nohup_prod_baselines.out 2>/dev/null && {
    echo "=== prod_keeper: relaunch A chain $(date '+%F %T') ===" >> nohup_prod_A.out
    setsid bash prod_chain_A.sh >> nohup_prod_A.out 2>&1 < /dev/null &
    exit 0
  }
}
pgrep -f "prod_cil_chain.sh|prod_baselines_chain.sh|prod_unlearn.py|prod_baselines.py|prod_eval.py" > /dev/null && exit 0
if [ ! -d adapters_prod/cil/epoch5 ] || ! python3 -c "import json;assert 'cil_ep5' in json.load(open('prod_results.json'))" 2>/dev/null; then
  echo "=== prod_keeper: relaunch CIL chain $(date '+%F %T') ===" >> nohup_prod_cil.out
  setsid bash prod_cil_chain.sh >> nohup_prod_cil.out 2>&1 < /dev/null &
else
  echo "=== prod_keeper: relaunch baselines chain $(date '+%F %T') ===" >> nohup_prod_baselines.out
  setsid bash prod_baselines_chain.sh >> nohup_prod_baselines.out 2>&1 < /dev/null &
fi
