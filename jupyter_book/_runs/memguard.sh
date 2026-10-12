#!/usr/bin/env bash
# Logs RSS of the C4/C5 jobs (FEMM ports 48017/48018 and their Python parents) every 60 s.
# Stops the C4/C5 queues if a FEMM process passes 2 GB or a Python job passes 3 GB
# (the C5 DQN Python sits at ~1.5 GB: torch CUDA context plus its replay buffer).
LOG=/home/akhan/code/compem_book/_runs/memguard.log
while true; do
  line="$(date '+%F %T')"; over=0
  for pid in $(pgrep -f "femm_port_fresh.py 4801[78]"); do
    rss=$(ps -o rss= -p $pid 2>/dev/null) || continue
    line="$line | py $pid $((rss/1024))MB"; [ "$rss" -gt 3145728 ] && over=1
  done
  for pid in $(pgrep -f "femm_headless --port 4801[78]"); do
    rss=$(ps -o rss= -p $pid 2>/dev/null) || continue
    line="$line | femm $pid $((rss/1024))MB"; [ "$rss" -gt 2097152 ] && over=1
  done
  line="$line | avail $(free -m | awk '/Mem:/{print $7}')MB swap_used $(free -m | awk '/Swap:/{print $3}')MB"
  echo "$line" >> $LOG
  if [ $over -eq 1 ]; then
    echo "$(date '+%F %T') OVER LIMIT: stopping C4/C5 queues" >> $LOG
    systemctl --user stop compem-queue-c4 compem-queue-c5 compem-queue-c4b
  fi
  sleep 60
done
