#!/bin/bash
# Start (or resume) the C4 tabular Q-learning run as a systemd user service.
# Safe to re-run after any interruption: if checkpoint.pkl exists it continues
# from the last finished episode, giving the same result as an uninterrupted run.
#   bash _runs/c4_resume/run_q_improved.sh            # start or resume
#   journalctl/log: _runs/data_logs/c4_q_improved_new.log
set -euo pipefail
B=/home/akhan/code/compem_book
OUT=$B/C4/designs/q_improved
LOG=$B/_runs/data_logs/c4_q_improved_new.log
UNIT=compem-c4-qlearn

if systemctl --user is-active -q $UNIT; then echo "$UNIT is already running"; exit 0; fi
systemctl --user reset-failed $UNIT 2>/dev/null || true

RESUME=()
[ -f $OUT/checkpoint.pkl ] && RESUME=(--resume) && echo "checkpoint found -> resuming"

systemd-run --user --unit=$UNIT --working-directory=$B/C4 \
  -p CPUQuota=200% -p Nice=10 -p StandardOutput=append:$LOG -p StandardError=append:$LOG \
  --setenv=PYTHONUNBUFFERED=1 --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  --setenv=PATH=/home/akhan/.local/bin:/usr/local/bin:/usr/bin:/bin \
  $B/.venv/bin/python $B/_runs/femm_port_fresh.py 48017 q_learn_improved \
  --results-dir $OUT "${RESUME[@]}"
echo "started $(date '+%F %T'); log: $LOG"
