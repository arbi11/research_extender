#!/usr/bin/env bash
# Runs the existing C3 data scripts (unchanged) on FEMM port 48016, alongside the C2 job on 48015.
cd /home/akhan/code/compem_book/C3
PY="../.venv/bin/python ../_runs/femm_port.py 48016"
run() { echo "=== $* $(date +%T)"; s=$(date +%s); "$@"; echo "rc=$? secs=$(( $(date +%s)-s ))"; }
run $PY data_gen.run_phase1
run $PY data_gen.run_phase2
run $PY data_gen.run_phase3
run $PY data_gen.run_phase4
# Pilot sweep to measure per-design time before committing to the full ~2800-design dataset_train sweep
run $PY data_gen.run_dataset --n 20 --seed 42 --dataset-root data_gen/outputs/dataset_train
echo C3_PILOT_DONE
