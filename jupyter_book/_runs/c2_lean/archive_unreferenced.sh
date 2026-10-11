#!/bin/bash
# Lean-up of C2: move every file nothing in the chapter references into an archive folder (reversible: plain mv).
# Run from anywhere:  bash _runs/c2_lean/archive_unreferenced.sh      (delete the archive later with rm -rf if you are sure)
set -e
cd /home/akhan/code/compem_book
A=_archive/C2_removed_2026-10-10
mkdir -p $A/models $A/dataset/pinn $A/dataset/grid_cache
cd C2
# my working backups of 04c / train_cnn + an old femm_solver copy
mv 04c_cnn_training.ipynb.orig 04c_cnn_training.ipynb.pre_label_fix 04c_cnn_training.ipynb.pre_cell11_fix 04c_cnn_training.ipynb.unexecuted train_cnn.py.orig femm_solver.py.femm42.bak ../$A/
# scripts / outputs no notebook uses (04a trains the ANN inline; cnn_model.py and uncertainty_utils.py are never imported)
mv c_core_40N_demo.py c_core_40N.png cnn_model.py uncertainty_utils.py train_c_core_ann.py performance_plots.png ../$A/
# legacy PINN script + its (584 % NMSE) results
mv pinns_train_final.py results ../$A/
# superseded by cnn_unet.py / pinn_base.py (the notebooks now import those)
mv train_cnn.py train_pinns.py ../$A/
# FEMM debug dumps written by femm_solver(debug_mode=True)
mv coil_debug.ans coil_debug.fem transformer_debug.ans transformer_debug.fem ../$A/
# train_cnn.py CLI outputs (04c trains its own baseline -> models/coil_baseline_unet_04c.pth)
mv models/coil_best_model.pth models/coil_history.pkl models/coil_predictions.png models/coil_summary.json models/coil_training_curves.png ../$A/models/
# duplicate generation checkpoint of the CNN dataset (1.17 GB) and the PINN obs-count variants nothing loads (906 MB)
mv dataset/coil_cnn_dataset_checkpoint.pkl ../$A/dataset/
mv dataset/pinn/coil_n50 dataset/pinn/coil_n500 ../$A/dataset/pinn/
# grid cache built for train_cnn.py's bounds (563 MB); 04c uses the other one
mv dataset/grid_cache/r256_e6bfba5c673a352d ../$A/dataset/grid_cache/
mv __pycache__ ../$A/ 2>/dev/null || true
# file-name case: _toc.yaml says content/01_intro
mv 01_Intro.ipynb 01_intro.ipynb
echo "archived to $A:"; du -sh ../$A; echo "C2 is now $(du -sh . | cut -f1)"
