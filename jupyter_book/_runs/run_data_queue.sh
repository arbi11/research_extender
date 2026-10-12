#!/usr/bin/env bash
# Usage: run_data_queue.sh <queue>   (c2 | c4 | c5)
# Runs the existing data scripts with the commands the notebooks document. C4/C5 go through
# _runs/femm_port_fresh.py on their own FEMM ports so they run alongside C2 (48015) and C3 (48016).
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
B=/home/akhan/code/compem_book; PY=$B/.venv/bin/python; L=$B/_runs/data_logs; mkdir -p $L
step() { local name=$1 dir=$2; shift 2
  [ -f $L/$name.done ] && { echo "skip $name"; return; }
  echo "=== $name $(date '+%F %T')"; s=$(date +%s)
  (cd $B/$dir && "$@") > $L/$name.log 2>&1; rc=$?
  echo "rc=$rc secs=$(( $(date +%s)-s )) $name"; [ $rc -eq 0 ] && touch $L/$name.done; }
case $1 in
c2)
  while pgrep -f generate_pinns_data.py >/dev/null; do sleep 20; done
  step c2_ann_data C2 $PY generate_ann_data.py ;;
c4)
  P="$PY $B/_runs/femm_port_fresh.py 48017"
  step c4_build_ccore C4 $P seqTO_env
  step c4_build_synrm C4 $P seqTO_SynRM_env --build-geometry --no-tests
  step c4_ga_ccore C4 $P ga_cCore
  step c4_ga_synrm C4 $P ga_synrm --pop 20 --n-gen 40 --stall 15
  step c4_q_synrm C4 $P q_learn_synrm --grid-size 7 --episodes 75 ;;
c5)
  P="$PY $B/_runs/femm_port_fresh.py 48018"
  while pgrep -f "run_data_queue.sh c4" >/dev/null; do sleep 60; done
  step c5_dqn C5 $P dqn_seqto --episodes 200 --reward-kind force --resume
  step c5_a2c C5 $P a2c_seqto ;;
esac
echo "QUEUE_$1_DONE"
