#!/bin/bash
cd /path/to/slice
# wait for the slice_dsB finisher to exit
while pgrep -f "finish_slice_dsb.sh" > /dev/null; do sleep 120; done
bash attacks_ablations.sh
