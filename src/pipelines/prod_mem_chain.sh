#!/bin/bash
cd /path/to/slice
export PYTORCH_ALLOC_CONF=expandable_segments:True
python3 -u continue_memorize.py && python3 -u prod_funnel.py
echo CHAIN_DONE
