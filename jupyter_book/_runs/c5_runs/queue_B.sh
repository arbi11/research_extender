#!/usr/bin/env bash
# Chapter 5 queue B (FEMM port 48020): A2C-v2, raw PPO, SB3-A2C.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
B=/home/akhan/code/compem_book; PY=$B/.venv/bin/python; L=$B/_runs/data_logs
P="$PY $B/_runs/femm_port_fresh.py 48020"
step() { local name=$1; shift
  [ -f $L/$name.done ] && { echo "skip $name"; return; }
  echo "=== $name $(date '+%F %T')"; s=$(date +%s)
  (cd $B/C5 && "$@") > $L/$name.log 2>&1; rc=$?
  echo "rc=$rc secs=$(( $(date +%s)-s )) $name"; [ $rc -eq 0 ] && touch $L/$name.done; }
step c5_a2c_v2 $P a2c_v2 --episodes 200
step c5_ppo_raw $P ppo_seqto --episodes 200
step c5_sb3_a2c $P sb3_runner --algo a2c --timesteps 6800 --reward-scale 0.001
echo QUEUE_c5B_DONE
