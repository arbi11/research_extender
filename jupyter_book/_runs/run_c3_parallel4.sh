#!/usr/bin/env bash
# Resume #3 of the C3 IPM sweep, capped at 2 FEMM workers (user: use at most 2 cores).
# Completed on disk at 08:48: 725 designs. The six remaining index gaps are split over 2 workers
# (~828 designs each); each worker runs its three gaps back to back. Fresh LHS seeds per invocation.
# Detached with setsid so a Claude session ending does not kill it.
cd /home/akhan/code/compem_book/C3
L=../_runs/c3_logs; mkdir -p $L
R=data_gen/outputs/dataset_train
run() { local port=$1 name=$2; shift 2
  echo "=== $name start $(date +%T)"; s=$(date +%s)
  ../.venv/bin/python ../_runs/femm_port.py $port data_gen.run_dataset "$@" --dataset-root $R > $L/$name.log 2>&1
  echo "rc=$? secs=$(( $(date +%s)-s )) $name"; }
( run 48016 ipm_X1a --n 331 --seed 61 --start 362;  run 48016 ipm_X1b --n 162 --seed 62 --start 879;  run 48016 ipm_X1c --n 336 --seed 63 --start 1240 ) &
( run 48019 ipm_X2a --n 327 --seed 64 --start 1594; run 48019 ipm_X2b --n 158 --seed 65 --start 2293; run 48019 ipm_X2c --n 341 --seed 66 --start 2460 ) &
wait
echo "=== rebuild masters $(date +%T)"
../.venv/bin/python -m data_gen.run_dataset --rebuild-master --dataset-root $R
../.venv/bin/python -m data_gen.run_dataset --rebuild-master --motor fscw --dataset-root data_gen/outputs/dataset_fscw
../.venv/bin/python -m data_gen.compute_pf_maps --dataset-root $R
echo C3_PARALLEL_DONE
