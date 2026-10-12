#!/usr/bin/env bash
# Resume #4 of the C3 IPM sweep, 2 FEMM workers (user cap: 2 cores).
# Completed on disk at 09:08: 744 designs. Six remaining index gaps split over 2 workers (~818 each),
# each worker runs its three gaps back to back. Fresh LHS seeds per invocation.
# Launched OUTSIDE the tool sandbox (previous launches were killed minutes after each session ended).
cd /home/akhan/code/compem_book/C3
L=../_runs/c3_logs; mkdir -p $L
R=data_gen/outputs/dataset_train
run() { local port=$1 name=$2; shift 2
  echo "=== $name start $(date +%T)"; s=$(date +%s)
  ../.venv/bin/python ../_runs/femm_port.py $port data_gen.run_dataset "$@" --dataset-root $R > $L/$name.log 2>&1
  echo "rc=$? secs=$(( $(date +%s)-s )) $name"; }
( run 48016 ipm_Y1a --n 321 --seed 67 --start 372;  run 48016 ipm_Y1b --n 162 --seed 68 --start 879;  run 48016 ipm_Y1c --n 336 --seed 69 --start 1240 ) &
( run 48019 ipm_Y2a --n 317 --seed 70 --start 1604; run 48019 ipm_Y2b --n 158 --seed 71 --start 2293; run 48019 ipm_Y2c --n 341 --seed 72 --start 2460 ) &
wait
echo "=== rebuild masters $(date +%T)"
../.venv/bin/python -m data_gen.run_dataset --rebuild-master --dataset-root $R
../.venv/bin/python -m data_gen.run_dataset --rebuild-master --motor fscw --dataset-root data_gen/outputs/dataset_fscw
../.venv/bin/python -m data_gen.compute_pf_maps --dataset-root $R
echo C3_PARALLEL_DONE
