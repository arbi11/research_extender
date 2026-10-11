#!/usr/bin/env bash
# C3 IPM sweep, memory-safe v2. Root cause of the OOM kills: data_gen.dataset_runner.aggregate() (the
# every-10-designs checkpoint) re-reads the whole master .npz once per existing design and keeps every copy
# alive -> memory ~ (designs in master)^2, ~39 GB at 1000 designs. Workaround without editing book code:
#   * each chunk of CH designs runs in a fresh python+FEMM against an EMPTY scratch dataset root
#     (its checkpoints then touch a <=CH-design master), with --start giving the true design numbering;
#   * finished design_NNN folders are moved into dataset_train after every chunk;
#   * at the end the master is rebuilt from the per-design params.npz files (master moved aside first so
#     the rebuild never opens a large existing master).
# 2 workers = 2 cores (user cap). RSS watchdog per chunk. Safe to re-run: resumes from folders on disk.
B=/home/akhan/code/compem_book; cd $B/C3
R=data_gen/outputs/dataset_train; L=$B/_runs/c3_logs; mkdir -p $L
SCR=$B/_runs/c3_chunk_roots; mkdir -p $SCR
CH=25; MAXMB=1500; SEED=${SEED0:-200}
frontier() { local hi=$(($1-1)) n
  for d in $R/design_*; do n=${d##*design_}; n=$((10#$n)); [ $n -ge $1 ] && [ $n -le $2 ] && [ $n -gt $hi ] && hi=$n; done
  echo $hi; }
worker() { local port=$1 name=$2; shift 2
  local seed=$((SEED + port))
  for range in "$@"; do
    local s=${range%:*} e=${range#*:}
    while :; do
      local next=$(( $(frontier $s $e) + 1 )); [ $next -gt $e ] && break
      local n=$(( e - next + 1 )); [ $n -gt $CH ] && n=$CH
      seed=$((seed+1)); local root=$SCR/${name}_$next; rm -rf $root; mkdir -p $root
      echo "=== $name chunk start=$next n=$n seed=$seed $(date +%T)"
      ../.venv/bin/python ../_runs/femm_port.py $port data_gen.run_dataset --n $n --seed $seed --start $next --dataset-root $root >> $L/$name.log 2>&1 &
      local pid=$!
      while [ -d /proc/$pid ]; do sleep 10
        local rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$pid/status 2>/dev/null)
        [ -n "$rss" ] && [ $rss -gt $MAXMB ] && { echo "!!! $name RSS ${rss}MB > ${MAXMB}MB, killing chunk"; kill $pid; }
      done
      wait $pid; local rc=$?
      local moved=0; for d in $root/design_*; do [ -d $d ] && mv -n $d $R/ && moved=$((moved+1)); done
      rm -rf $root
      echo "rc=$rc $name chunk start=$next moved=$moved ok=$(ls $R/design_*/params.npz 2>/dev/null | wc -l)"
      pkill -f "femm_headless --port $port\$"; sleep 1
    done
  done
  echo "=== $name RANGES_DONE $(date +%T)"; }
worker 48016 ipm_V1 372:692 879:1040 1240:1575 &
worker 48019 ipm_V2 1604:1920 2293:2450 2460:2800 &
wait
echo "=== rebuild masters $(date +%T)"
mv -f $R/master_dataset.npz $R/master_dataset_checkpoint_old.npz 2>/dev/null
../.venv/bin/python -m data_gen.run_dataset --rebuild-master --dataset-root $R
../.venv/bin/python -m data_gen.run_dataset --rebuild-master --motor fscw --dataset-root data_gen/outputs/dataset_fscw
../.venv/bin/python -m data_gen.compute_pf_maps --dataset-root $R
echo C3_PARALLEL_DONE
