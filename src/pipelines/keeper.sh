#!/bin/bash
# Cron keeper: restart the attacks pipeline if it died and no work is running.
cd /path/to/slice
[ -f ATTACKS_DONE ] && exit 0
pgrep -f "attacks_ablations.sh" > /dev/null && exit 0
pgrep -f "eval_lc.py|unlearn_lc.py|paraphrase_probe.py|relearn_lc.py" > /dev/null && exit 0
echo "=== keeper relaunch $(date '+%F %T') ===" >> nohup_attacks.out
bash attacks_ablations.sh >> nohup_attacks.out 2>&1
