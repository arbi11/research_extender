#!/usr/bin/env bash
# Resume #2 of the C3 IPM sweep, capped at 4 FEMM workers (user: use at most 4 of the 8 cores).
# Completed on disk at 08:40: 694 designs. The six remaining index gaps are rebalanced over 4 workers
# (~508 designs each); a worker runs its two gaps back to back. Fresh LHS seeds per invocation.
# Detached with setsid so a Claude session ending does not kill it.
cd /home/akhan/code/compem_book/C3
L=../_runs/c3_logs; mkdir -p $L
R=data_gen/outputs/dataset_train
run() { local port=$1 name=$2; shift 2
  echo "=== $name start $(date +%T)"; s=$(date +%s)
  ../.venv/bin/python ../_runs/femm_port.py $port data_gen.run_dataset "$@" --dataset-root $R > $L/$name.log 2>&1
  echo "rc=$? secs=$(( $(date +%s)-s )) $name"; }
( run 48016 ipm_W1a --n 341 --seed 53 --start 352;  run 48016 ipm_W1b --n 167 --seed 54 --start 702  ) &
( run 48019 ipm_W2a --n 172 --seed 55 --start 869;  run 48019 ipm_W2b --n 336 --seed 56 --start 1240 ) &
( run 48020 ipm_W3a --n 335 --seed 57 --start 1586; run 48020 ipm_W3b --n 173 --seed 58 --start 2110 ) &
( run 48022 ipm_W4a --n 168 --seed 59 --start 2283; run 48022 ipm_W4b --n 341 --seed 60 --start 2460 ) &
wait
echo "=== rebuild masters $(date +%T)"
../.venv/bin/python -m data_gen.run_dataset --rebuild-master --dataset-root $R
../.venv/bin/python -m data_gen.run_dataset --rebuild-master --motor fscw --dataset-root data_gen/outputs/dataset_fscw
../.venv/bin/python -m data_gen.compute_pf_maps --dataset-root $R
echo C3_PARALLEL_DONE
