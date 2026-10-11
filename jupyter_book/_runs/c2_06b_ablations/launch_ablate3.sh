#!/bin/bash
cd /home/akhan/code/compem_book
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
S=/tmp/claude-1000/-home-akhan-code-compem-book/b1b56765-ee69-4156-8f74-4e554b215093/scratchpad
.venv/bin/python -I $S/ablate_06b_plain.py E 3000 > _runs/data_logs/c2_06b_ablation_E.log 2>&1 &
.venv/bin/python -I $S/ablate_06b_plain.py F 3000 > _runs/data_logs/c2_06b_ablation_F.log 2>&1 &
wait; echo "ABLATION3_DONE" >> _runs/data_logs/c2_06b_ablation_E.log
