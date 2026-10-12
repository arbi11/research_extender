#!/bin/bash
cd /home/akhan/code/compem_book
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
S=/tmp/claude-1000/-home-akhan-code-compem-book/b1b56765-ee69-4156-8f74-4e554b215093/scratchpad
.venv/bin/python -I $S/ablate_06b.py A 1500 > _runs/data_logs/c2_06b_ablation_A.log 2>&1 &
.venv/bin/python -I $S/ablate_06b.py B 1500 > _runs/data_logs/c2_06b_ablation_B.log 2>&1 &
wait
echo "ABLATION_DONE" >> _runs/data_logs/c2_06b_ablation_A.log
