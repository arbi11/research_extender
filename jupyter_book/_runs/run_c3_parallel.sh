#!/usr/bin/env bash
# C3 data in parallel (existing scripts, unchanged; FEMM shims via _runs/femm_port.py).
#   IPM dataset_train: designs 1-160 done by the single sweep (run_c3_sweep.sh, stopped 00:09);
#   161-2800 split over 3 workers, each with its own FEMM port, LHS seed and disjoint design_NNN range.
#   FSCW dataset_fscw: 60 LHS designs (book: 49 valid).
#   Workers race on master_dataset.npz checkpoints, so masters are rebuilt from design_NNN/params.npz at the end,
#   then the power-factor master (no FEMM) is derived from dataset_train.
cd /home/akhan/code/compem_book/C3
L=../_runs/c3_logs; mkdir -p $L
W() { local port=$1 name=$2; shift 2
  ( s=$(date +%s); echo "=== $name start $(date +%T)"
    ../.venv/bin/python ../_runs/femm_port.py $port data_gen.run_dataset "$@" > $L/$name.log 2>&1
    echo "rc=$? secs=$(( $(date +%s)-s )) $name" ) & }
W 48016 ipm_A --n 880 --seed 46 --start 161  --dataset-root data_gen/outputs/dataset_train
W 48019 ipm_B --n 880 --seed 44 --start 1041 --dataset-root data_gen/outputs/dataset_train
W 48020 ipm_C --n 880 --seed 45 --start 1921 --dataset-root data_gen/outputs/dataset_train
W 48022 fscw  --n 60  --seed 42 --start 1 --motor fscw --dataset-root data_gen/outputs/dataset_fscw
wait
echo "=== rebuild masters $(date +%T)"
../.venv/bin/python -m data_gen.run_dataset --rebuild-master --dataset-root data_gen/outputs/dataset_train
../.venv/bin/python -m data_gen.run_dataset --rebuild-master --motor fscw --dataset-root data_gen/outputs/dataset_fscw
../.venv/bin/python -m data_gen.compute_pf_maps --dataset-root data_gen/outputs/dataset_train
echo C3_PARALLEL_DONE
