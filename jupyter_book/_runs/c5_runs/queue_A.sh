#!/usr/bin/env bash
# Chapter 5 queue A (FEMM port 48019): env probes, PPO-v2, SB3-PPO, random-policy probe, cell-flips.
# Each step runs the book script unchanged through _runs/femm_port_fresh.py; .done markers in _runs/data_logs/.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
B=/home/akhan/code/compem_book; PY=$B/.venv/bin/python; L=$B/_runs/data_logs
P="$PY $B/_runs/femm_port_fresh.py 48019"
step() { local name=$1; shift
  [ -f $L/$name.done ] && { echo "skip $name"; return; }
  echo "=== $name $(date '+%F %T')"; s=$(date +%s)
  (cd $B/C5 && "$@") > $L/$name.log 2>&1; rc=$?
  echo "rc=$rc secs=$(( $(date +%s)-s )) $name"; [ $rc -eq 0 ] && touch $L/$name.done; }
step c5_probe_determinism $P env_probe determinism --n 5
step c5_probe_random_search $P env_probe random-search --n 200 --densities 0.1 0.3 0.5 0.7
step c5_probe_action_effect $P env_probe action-effect --reps 25
step c5_ppo_v2 $P ppo_v2 --episodes 200
step c5_sb3_ppo $P sb3_runner --algo ppo --timesteps 6800 --reward-scale 0.001
step c5_probe_random_policy $P env_probe random-policy --episodes 50
step c5_probe_cell_flips $P env_probe cell-flips --from-best designs/dqn_run/best_mask.npy
echo QUEUE_c5A_DONE
