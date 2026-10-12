#!/usr/bin/env bash
# C2 continuation: wait for the CNN data run, then PINN data, model training, and notebook re-runs.
# All scripts are the unchanged Notion code, run with their own defaults.
cd /home/akhan/code/compem_book/C2
PY=../.venv/bin/python
run() { echo "=== $* $(date +%T)"; s=$(date +%s); "$@"; echo "rc=$? secs=$(( $(date +%s)-s ))"; }
while pgrep -f generate_cnn_training_data.py >/dev/null; do sleep 20; done
run $PY generate_pinns_data.py --problem-type coil
run $PY train_c_core_ann.py
run $PY train_cnn.py --problem-type coil
run $PY train_pinns.py --problem-type coil
run $PY pinns_train_final.py
cd ..
for nb in C2/03_data_pipeline C2/04a_ann_fundamentals C2/04c_cnn_training C2/05_uncertainty_quantification C2/06a_pinn_theory_formulation C2/06b_pinns_methodology; do
  run .venv/bin/jupyter nbconvert --to notebook --execute --allow-errors --ExecutePreprocessor.timeout=-1 \
     --output-dir _runs/executed --output "${nb//\//__}" "$nb.ipynb"
done
echo C2_DONE
