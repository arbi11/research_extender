#!/usr/bin/env bash
# Resume of run_c3_parallel.sh (whose 3 IPM workers died at 01:14 when the Claude session ended).
# Completed on disk at restart: design_1-343, 1041-1229, 1921-2100 (642 designs) + all 60 FSCW.
# The three remaining ranges are each split in two (6 workers, 6 FEMM ports) since the box is idle.
# New LHS seeds so the resumed samples don't duplicate the earlier seeds' first N points.
# Launched with setsid so a Claude session ending no longer kills the sweep.
cd /home/akhan/code/compem_book/C3
L=../_runs/c3_logs; mkdir -p $L
W() { local port=$1 name=$2; shift 2
  ( s=$(date +%s); echo "=== $name start $(date +%T)"
    ../.venv/bin/python ../_runs/femm_port.py $port data_gen.run_dataset "$@" > $L/$name.log 2>&1
    echo "rc=$? secs=$(( $(date +%s)-s )) $name" ) & }
R=data_gen/outputs/dataset_train
W 48016 ipm_A1 --n 349 --seed 47 --start 344  --dataset-root $R
W 48019 ipm_A2 --n 348 --seed 48 --start 693  --dataset-root $R
W 48020 ipm_B1 --n 346 --seed 49 --start 1230 --dataset-root $R
W 48022 ipm_B2 --n 345 --seed 50 --start 1576 --dataset-root $R
W 48023 ipm_C1 --n 350 --seed 51 --start 2101 --dataset-root $R
W 48024 ipm_C2 --n 350 --seed 52 --start 2451 --dataset-root $R
wait
echo "=== rebuild masters $(date +%T)"
../.venv/bin/python -m data_gen.run_dataset --rebuild-master --dataset-root $R
../.venv/bin/python -m data_gen.run_dataset --rebuild-master --motor fscw --dataset-root data_gen/outputs/dataset_fscw
../.venv/bin/python -m data_gen.compute_pf_maps --dataset-root $R
echo C3_PARALLEL_DONE
