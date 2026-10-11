#!/usr/bin/env bash
# C3 IPM sweep, memory-safe: each worker runs data_gen.run_dataset in chunks of CH designs in a FRESH
# python + femm_headless process (the worker leaks ~13-80 MB per design and was OOM-killed at 7-12 GB
# five times on 2026-10-10). 2 workers = 2 cores (user cap). A watchdog kills a chunk whose python
# RSS exceeds MAXMB; the loop then resumes from the highest design dir already written.
# Book code unchanged; FEMM shims via _runs/femm_port.py. Safe to re-run: resumes from what is on disk.
B=/home/akhan/code/compem_book; cd $B/C3
R=data_gen/outputs/dataset_train; L=$B/_runs/c3_logs; mkdir -p $L
CH=15; MAXMB=1500; SEED=${SEED0:-100}
frontier() { # highest existing design_NNN dir index in [$1,$2], or $1-1
  local hi=$(($1-1)) n
  for d in $R/design_*; do n=${d##*design_}; n=$((10#$n)); [ $n -ge $1 ] && [ $n -le $2 ] && [ $n -gt $hi ] && hi=$n; done
  echo $hi; }
worker() { local port=$1 name=$2; shift 2   # remaining args: start:end ranges
  local seed=$((SEED + port))
  for range in "$@"; do
    local s=${range%:*} e=${range#*:}
    while :; do
      local next=$(( $(frontier $s $e) + 1 )); [ $next -gt $e ] && break
      local n=$(( e - next + 1 )); [ $n -gt $CH ] && n=$CH
      seed=$((seed+1))
      echo "=== $name chunk start=$next n=$n seed=$seed $(date +%T)"
      ../.venv/bin/python ../_runs/femm_port.py $port data_gen.run_dataset --n $n --seed $seed --start $next --dataset-root $R >> $L/$name.log 2>&1 &
      local pid=$!
      while [ -d /proc/$pid ]; do
        sleep 10
        local rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$pid/status 2>/dev/null)
        [ -n "$rss" ] && [ $rss -gt $MAXMB ] && { echo "!!! $name RSS ${rss}MB > ${MAXMB}MB, killing chunk"; kill $pid; }
      done
      wait $pid; echo "rc=$? $name chunk start=$next"
      pkill -f "femm_headless --port $port\$"; sleep 1
    done
  done
  echo "=== $name RANGES_DONE $(date +%T)"; }
worker 48016 ipm_Z1 372:692 879:1040 1240:1575 &
worker 48019 ipm_Z2 1604:1920 2293:2450 2460:2800 &
wait
echo "=== rebuild masters $(date +%T)"
../.venv/bin/python -m data_gen.run_dataset --rebuild-master --dataset-root $R
../.venv/bin/python -m data_gen.run_dataset --rebuild-master --motor fscw --dataset-root data_gen/outputs/dataset_fscw
../.venv/bin/python -m data_gen.compute_pf_maps --dataset-root $R
echo C3_PARALLEL_DONE
