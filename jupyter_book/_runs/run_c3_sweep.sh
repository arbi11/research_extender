#!/usr/bin/env bash
# Full C3 dataset_train sweep (existing script, unchanged): 2780 designs on top of the 20-design pilot = 2800 sampled, as in the book.
cd /home/akhan/code/compem_book/C3
echo "=== sweep start $(date +%T)"; s=$(date +%s)
../.venv/bin/python ../_runs/femm_port.py 48016 data_gen.run_dataset --n 2780 --seed 43 --start -1 --dataset-root data_gen/outputs/dataset_train
echo "rc=$? secs=$(( $(date +%s)-s ))"
echo C3_SWEEP_DONE
