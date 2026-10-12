#!/bin/bash
cd /home/akhan/code/compem_book
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
S=/tmp/claude-1000/-home-akhan-code-compem-book/b1b56765-ee69-4156-8f74-4e554b215093/scratchpad
L=_runs/data_logs
.venv/bin/python -I $S/ablate_06b_v3.py G 0.02 3000 > $L/c2_06b_ablation_G002.log 2>&1 &
.venv/bin/python -I $S/ablate_06b_v3.py G 1 3000 > $L/c2_06b_ablation_G1.log 2>&1 &
wait
.venv/bin/python -I $S/ablate_06b_v3.py H 1 3000 > $L/c2_06b_ablation_H1.log 2>&1 &
.venv/bin/python -I $S/ablate_06b_v3.py H 0.02 3000 > $L/c2_06b_ablation_H002.log 2>&1 &
wait; echo "ABLATION4_DONE" >> $L/c2_06b_ablation_G002.log
