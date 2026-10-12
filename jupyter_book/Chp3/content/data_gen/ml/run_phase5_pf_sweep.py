"""Phase 5C sweep: sample-efficiency curve for PF transfer learning.

Loops over (n_train, seed, mode) and writes a combined `sweep_metrics.csv` that
the Chapter 5 notebook reads to plot the transfer-learning headline figure.

Modes:
    scratch              random init, full training.
    pretrained-naive     load IPM eta MapDecoder weights, fine-tune everything.
    pretrained-reset     load IPM eta weights, re-init the final output ConvTranspose
                         (breaks the eta output-distribution bias), fine-tune.

Usage:
    python -m data_gen.ml.run_phase5_pf_sweep
    python -m data_gen.ml.run_phase5_pf_sweep --pretrained data_gen/outputs/phase5_map_train/weights_fold1.pt
"""
from __future__ import annotations
import argparse
import csv
import logging
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA_GEN_ROOT = HERE.parent
DEFAULT_OUT_DIR = DATA_GEN_ROOT / "outputs" / "phase5_pf_train"
DEFAULT_DATASET = DATA_GEN_ROOT / "outputs" / "dataset_train" / "master_pf_dataset.npz"
DEFAULT_PRETRAINED = DATA_GEN_ROOT / "outputs" / "phase5_map_train" / "weights_fold1.pt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--n-train-sizes", type=int, nargs="+",
                        default=[10, 25, 50, 100, 200, 400])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--modes", type=str, nargs="+",
                        default=["scratch", "pretrained-naive", "pretrained-reset"],
                        choices=["scratch", "pretrained-naive", "pretrained-reset"])
    parser.add_argument("--arch-preset", type=str, default="baseline",
                        help="MapDecoder architecture preset to use throughout "
                             "the sweep. Must match the architecture of the "
                             "--pretrained checkpoint (default: baseline).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s")
    log = logging.getLogger("sweep")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = args.out_dir / "sweep_metrics.csv"
    log.info("Output: %s", sweep_csv)

    runs = [(n, s, m) for n in args.n_train_sizes for s in args.seeds for m in args.modes]
    log.info("Sweep size: %d runs (%d sizes x %d seeds x %d modes)",
             len(runs), len(args.n_train_sizes), len(args.seeds), len(args.modes))

    rows = []
    t0 = time.time()
    for k, (n_train, seed, mode) in enumerate(runs, 1):
        cmd = [
            sys.executable, "-m", "data_gen.ml.run_phase5_pf",
            "--dataset", str(args.dataset),
            "--out-dir", str(args.out_dir),
            "--n-train", str(n_train),
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--arch-preset", args.arch_preset,
            "--quiet",
        ]
        if mode != "scratch":
            cmd += ["--pretrained", str(args.pretrained)]
            if mode == "pretrained-reset":
                cmd += ["--reset-output"]

        t_start = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t_start
        if r.returncode != 0:
            log.error("Run failed: n_train=%d seed=%d mode=%s\nstderr: %s",
                      n_train, seed, mode, r.stderr[-500:])
            continue

        # Read the per-run metrics.csv.
        if mode == "scratch":
            run_id = f"n{n_train:04d}_s{seed}_scratch"
        else:
            tag = "pretrained-reset" if mode == "pretrained-reset" else "pretrained-naive"
            run_id = f"n{n_train:04d}_s{seed}_{tag}"
        metrics_path = args.out_dir / run_id / "metrics.csv"
        if not metrics_path.exists():
            log.warning("metrics.csv missing for %s", run_id)
            continue
        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        log.info("  [%2d/%d] n_train=%4d  seed=%d  mode=%-18s  R2=%6s  (%.1f s)",
                 k, len(runs), n_train, seed, mode,
                 rows[-1]["val_r2"][:6] if rows else "n/a", elapsed)

    # Combine and write.
    if not rows:
        log.error("No successful runs to write")
        return 2
    fieldnames = list(rows[0].keys())
    with open(sweep_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("Wrote %s (%d runs)", sweep_csv, len(rows))
    log.info("Total elapsed: %.1f min", (time.time() - t0) / 60.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
