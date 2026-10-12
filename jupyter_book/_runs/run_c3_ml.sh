#!/usr/bin/env bash
# C3 phase-5 model training + notebook re-run. Waits for the dataset sweep (run_c3_parallel2.sh) to finish.
# Commands are the ones documented in C3/04 and C3/05 (book code unchanged; train.py pins device=cpu).
# Each step is skipped if its .done marker exists, so the script can be re-run after an interruption.
# CPU cap (user): at most 2 cores -> one process at a time, 2 threads.
B=/home/akhan/code/compem_book; PY=$B/.venv/bin/python; L=$B/_runs/c3_ml_logs; mkdir -p $L
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
until grep -q C3_PARALLEL_DONE $B/_runs/run_c3_chunked2.log 2>/dev/null; do sleep 60; done
cd $B/C3
O=data_gen/outputs; DS=$O/dataset_train/master_dataset.npz; PF=$O/dataset_train/master_pf_dataset.npz; FS=$O/dataset_fscw/master_dataset.npz
MAXMB=3500   # memory watchdog: a step whose RSS exceeds this is killed (rc!=0, no .done) rather than swapping the box
step() { local name=$1; shift
  [ -f $L/$name.done ] && { echo "skip $name"; return; }
  echo "=== $name $(date '+%F %T')"; s=$(date +%s)
  "$@" > $L/$name.log 2>&1 & local pid=$!
  while [ -d /proc/$pid ]; do sleep 20
    local rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$pid/status 2>/dev/null)
    [ -n "$rss" ] && [ $rss -gt $MAXMB ] && { echo "!!! $name RSS ${rss}MB > ${MAXMB}MB, killing"; kill $pid; }
  done
  wait $pid; rc=$?
  echo "rc=$rc secs=$(( $(date +%s)-s )) $name"; [ $rc -eq 0 ] && touch $L/$name.done; }
M="$PY -m data_gen.ml"

# All sequential (2-core cap). Order: what C3/04 needs first, then the arch variants, then C3/05's transfer runs.
step map_train      $M.run_phase5_map      --dataset $DS --out-dir $O/phase5_map_train --epochs 1500 --folds 5 --weight-decay 1e-4
step envelope_train $M.run_phase5_envelope --dataset $DS --out-dir $O/phase5_envelope_train --epochs 1500 --folds 5
for p in width4x width2x depth1; do
  step map_train_arch_$p $M.run_phase5_map --dataset $DS --out-dir $O/phase5_map_train_arch/$p --arch-preset $p --epochs 1500 --folds 5 --weight-decay 1e-4
done
step map_fscw        $M.run_phase5_map --dataset $FS --out-dir $O/phase5_map_fscw_train --folds 5 --epochs 1500 --weight-decay 1e-4
step map_fscw_w4x    $M.run_phase5_map --dataset $FS --out-dir $O/phase5_map_fscw_train_width4x --arch-preset width4x --folds 5 --epochs 1500 --weight-decay 1e-4
step pf_sweep        $M.run_phase5_pf_sweep --dataset $PF --out-dir $O/phase5_pf_train --pretrained $O/phase5_map_train/weights_fold1.pt --epochs 1500
step pf_demo_reset   $M.run_phase5_pf --dataset $PF --out-dir $O/phase5_pf_train --n-train 50 --seed 0 --pretrained $O/phase5_map_train/weights_fold1.pt --reset-output --freeze head
step pf_sweep_w4x    $M.run_phase5_pf_sweep --dataset $PF --out-dir $O/phase5_pf_train_width4x --pretrained $O/phase5_map_train_arch/width4x/weights_fold1.pt --arch-preset width4x --epochs 1500
step fscw_sweep      $M.run_phase5_fscw_sweep --dataset $FS --out-dir $O/phase5_fscw_train --pretrained $O/phase5_map_train/weights_fold1.pt --epochs 1500
step fscw_sweep_w4x  $M.run_phase5_fscw_sweep --dataset $FS --out-dir $O/phase5_fscw_train_width4x --pretrained $O/phase5_map_train_arch/width4x/weights_fold1.pt --arch-preset width4x --epochs 1500
# Re-execute the C3 notebooks (same nbconvert settings as _runs/run_all.sh)
cd $B
for nb in C3/[0-9]*.ipynb; do
  step nb_$(basename $nb .ipynb) $B/.venv/bin/jupyter nbconvert --to notebook --execute --allow-errors \
     --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=python3 \
     --output-dir $B/_runs/executed --output "${nb//\//__}" "$nb"
done
echo C3_ML_DONE
