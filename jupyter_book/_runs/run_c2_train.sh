#!/usr/bin/env bash
# C2 model training on the GPU with the unchanged Notion training scripts and their defaults.
export PYTHONUNBUFFERED=1
B=/home/akhan/code/compem_book; PY=$B/.venv/bin/python; L=$B/_runs/data_logs
step() { local name=$1; shift
  [ -f $L/$name.done ] && { echo "skip $name"; return; }
  echo "=== $name $(date '+%F %T')"; s=$(date +%s)
  (cd $B/C2 && "$@") > $L/$name.log 2>&1; rc=$?
  echo "rc=$rc secs=$(( $(date +%s)-s )) $name"; [ $rc -eq 0 ] && touch $L/$name.done; }
step c2_train_ann   $PY train_c_core_ann.py
step c2_train_cnn   $PY train_cnn.py --problem-type coil
step c2_train_pinns $PY train_pinns.py --problem-type coil
step c2_pinns_final $PY pinns_train_final.py
echo C2_TRAIN_DONE
