#!/usr/bin/env bash
# Runs the two reconstructed C4 scripts, one FEMM job at a time: waits for the C5 queue
# (DQN, then A2C) to finish, then q_learn_improved.py, then diagnose_synrm.py if present.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
B=/home/akhan/code/compem_book; PY=$B/.venv/bin/python; L=$B/_runs/data_logs
P="$PY $B/_runs/femm_port_fresh.py 48017"
step() { local name=$1; shift
  [ -f $L/$name.done ] && { echo "skip $name"; return; }
  echo "=== $name $(date '+%F %T')"; s=$(date +%s)
  (cd $B/C4 && "$@") > $L/$name.log 2>&1; rc=$?
  echo "rc=$rc secs=$(( $(date +%s)-s )) $name"; [ $rc -eq 0 ] && touch $L/$name.done; }
while systemctl --user is-active -q compem-queue-c5; do sleep 60; done
step c4_q_improved $P q_learn_improved
[ -f $B/C4/diagnose_synrm.py ] && step c4_diag_synrm $P diagnose_synrm
echo QUEUE_c4b_DONE
