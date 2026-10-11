#!/usr/bin/env bash
# Runs the existing C2 data scripts (unchanged) with their defaults, then re-runs the C2 notebooks.
cd /home/akhan/code/compem_book/C2
PY=../.venv/bin/python
run() { echo "=== $* $(date +%T)"; s=$(date +%s); "$@"; echo "rc=$? secs=$(( $(date +%s)-s ))"; }
run $PY generate_ann_data.py
run $PY generate_cnn_training_data.py --problem-type coil
run $PY generate_pinns_data.py --problem-type coil
cd ..
for nb in C2/03_data_pipeline C2/04a_ann_fundamentals C2/04c_cnn_training C2/06a_pinn_theory_formulation C2/06b_pinns_methodology; do
  run .venv/bin/jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=7200 \
     --output-dir _runs/executed --output "${nb//\//__}" "$nb.ipynb"
done
echo C2_DONE
