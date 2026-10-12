#!/bin/bash
cd /home/akhan/code/compem_book
until grep -q SMOKE_RC _runs/data_logs/c2_04c_smoke_v2.log 2>/dev/null; do sleep 15; done
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
S=/tmp/claude-1000/-home-akhan-code-compem-book/b1b56765-ee69-4156-8f74-4e554b215093/scratchpad
.venv/bin/python -I $S/ablate_06b_plain.py C 3000 > _runs/data_logs/c2_06b_ablation_C.log 2>&1 &
.venv/bin/python -I $S/ablate_06b_plain.py D 3000 > _runs/data_logs/c2_06b_ablation_D.log 2>&1 &
wait; echo "ABLATION2_DONE" >> _runs/data_logs/c2_06b_ablation_C.log
