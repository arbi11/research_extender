#!/usr/bin/env bash
# pinns_train_final.py trains fine on the GPU but its final plot mixes CPU and GPU tensors
# (a bug in the Notion code). Re-run it on CPU (2 threads) after the GPU queue, no code changes.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
B=/home/akhan/code/compem_book
while pgrep -f "run_c2_train2.s[h]" >/dev/null; do sleep 30; done
echo "=== c2_pinns_final_cpu $(date '+%F %T')"; s=$(date +%s)
(cd $B/C2 && $B/.venv/bin/python pinns_train_final.py) > $B/_runs/data_logs/c2_pinns_final_cpu.log 2>&1; rc=$?
echo "rc=$rc secs=$(( $(date +%s)-s ))"; [ $rc -eq 0 ] && touch $B/_runs/data_logs/c2_pinns_final.done
echo C2_TRAIN3_DONE
