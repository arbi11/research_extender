#!/bin/bash
# Headless run of notebook 04c U-Net: args = epochs, variant (dense/dilated) (see run_04c_dense.py). Timestamps each log line.
B=/home/akhan/code/compem_book
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$B/C2" || exit 1
echo "start $(date '+%F %T')"
/usr/bin/time -v "$B/.venv/bin/python" "$B/_runs/c2_04c_dense/run_04c_dense.py" --epochs "${1:-50}" --variant "${2:-dense}" 2>&1 \
  | while IFS= read -r line; do printf '%s %s\n' "$(date +%H:%M:%S)" "$line"; done
echo "rc=${PIPESTATUS[0]} end $(date '+%F %T')"
