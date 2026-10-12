#!/usr/bin/env bash
# Logs RSS of the Chapter 5 jobs on FEMM ports 48019/48020 every 5 min (log only, no action).
while pgrep -f "femm_port_fresh.py 4801[9]|femm_port_fresh.py 4802[0]" >/dev/null; do
  line="$(date '+%F %T') avail=$(free -m | awk '/Mem/{print $7}')MB"
  for pid in $(pgrep -f "femm_port_fresh.py 4801[9]|femm_port_fresh.py 4802[0]"); do
    rss=$(ps -o rss= -p $pid 2>/dev/null) || continue; line="$line | py $pid $((rss/1024))MB"
  done
  echo "$line"; sleep 300
done
echo "$(date '+%F %T') no C5 jobs left"
