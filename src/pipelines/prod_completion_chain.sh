#!/bin/bash
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
# run after the extras chain finishes
while ! grep -q PROD_EXTRAS_DONE nohup_prod_extras.out 2>/dev/null; do sleep 300; done
MEM=adapters_prod/memorized/epoch10
have () { python3 -c "import json;assert 'humaneval_completion' in json.load(open('utility_results.json')).get('$1',{})" 2>/dev/null; }
have prodtask_memorized || python3 -u prod_completion_util.py --tag prodtask_memorized --adapters $MEM
have prodtask_slice       || python3 -u prod_completion_util.py --tag prodtask_slice --adapters $MEM,adapters_prod/slice/epoch5
have prodtask_prod      || python3 -u prod_completion_util.py --tag prodtask_prod --adapters $MEM,adapters_prod/prod/epoch3
echo PROD_COMPLETION_DONE
