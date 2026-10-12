#!/usr/bin/env bash
# Setup/test harness only: syncs the uv env, then executes every notebook unchanged.
cd /home/akhan/code/compem_book
export UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=10 UV_CONCURRENT_DOWNLOADS=4
UV=~/.local/bin/uv
until $UV sync; do echo "sync failed, retrying in 30s"; sleep 30; done
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
mkdir -p _runs/executed
for nb in AppA/[0-9]*.ipynb C1/[0-9]*.ipynb C2/[0-9]*.ipynb C3/[0-9]*.ipynb C4/[0-9]*.ipynb C5/[0-9]*.ipynb; do
  out="_runs/executed/${nb//\//__}"
  [ -f "$out.done" ] && continue
  echo "=== $nb $(date +%T)"
  start=$(date +%s)
  .venv/bin/jupyter nbconvert --to notebook --execute --allow-errors \
     --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3 \
     --output-dir "$(pwd)/_runs/executed" --output "${nb//\//__}" "$nb" > "$out.log" 2>&1
  echo "rc=$? secs=$(( $(date +%s)-start ))" | tee -a "$out.log"
  touch "$out.done"
done
echo ALL_DONE
