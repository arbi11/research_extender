"""Multi-design dataset driver.

Generates a Latin-Hypercube sample of motor designs, runs the 4-phase
data-collection pipeline on each, and appends them to
outputs/dataset/master_dataset.npz.

Usage (Windows, native Python with pyFEMM + FEMM .exe installed):

    # Fresh sweep starting from design_001:
    python -m data_gen.run_dataset --n 300 --seed 42

    # Extend an existing sweep: pick up one past the highest design_NNN folder.
    # New samples use a different LHS seed so they don't duplicate the prior coverage.
    python -m data_gen.run_dataset --n 2700 --seed 43 --start -1

    # Or start at an explicit index (writes design_188.. when --start 188):
    python -m data_gen.run_dataset --n 2700 --seed 43 --start 188

The master_dataset.npz is **additive**: existing designs are preserved and the
new run's results are merged in. Dedup is by design_idx (incoming overrides).

Per-design wall time: ~30-60 s (Phase 1 9 solves + Phase 3 30 solves).
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

from .dataset_runner import rebuild_master_from_disk, run_dataset


HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
DEFAULT_DATASET_ROOT = HERE / "outputs" / "dataset"


def _default_dataset_root(motor_type: str) -> Path:
    return HERE / "outputs" / ("dataset" if motor_type == "ipm" else f"dataset_{motor_type}")


def _setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"dataset_{ts}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    return log_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="number of designs to sample")
    parser.add_argument("--seed", type=int, default=42, help="LHS seed")
    parser.add_argument("--start", type=int, default=1,
                        help="Starting design index. Use -1 to auto-detect: "
                             "scans dataset folder and starts one past the "
                             "highest existing design_NNN.")
    parser.add_argument("--phase3-speeds", type=int, default=6,
                        help="Phase-3 grid speed count per design")
    parser.add_argument("--phase3-torques", type=int, default=5,
                        help="Phase-3 grid torque count per design")
    parser.add_argument("--rebuild-master", action="store_true",
                        help="Don't run any FEMM. Scan design_NNN/params.npz "
                             "files and reconstruct master_dataset.npz. "
                             "Use this after parallel runs or if a checkpoint "
                             "race corrupted the master.")
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help="Directory containing design_NNN/ subfolders and "
                             "master_dataset.npz. Defaults to outputs/dataset "
                             "(IPM) or outputs/dataset_<motor> for other motor "
                             "types. Use outputs/dataset_train to isolate "
                             "training from the live data-generation folder.")
    parser.add_argument("--motor", type=str, default="ipm",
                        choices=["ipm", "fscw"],
                        help="Motor topology to sweep. Selects geometry and "
                             "design-space sampler. IPM is the original 12s/4p "
                             "interior-PM machine; FSCW is the 12s/10p "
                             "surface-PM concentrated-winding machine.")
    args = parser.parse_args()
    if args.dataset_root is None:
        args.dataset_root = _default_dataset_root(args.motor)

    log_path = _setup_logging()
    log = logging.getLogger("run_dataset")
    dataset_root = args.dataset_root

    if args.rebuild_master:
        log.info("=== Rebuilding master_dataset.npz from disk (motor=%s) ===",
                 args.motor)
        log.info("Log file: %s", log_path)
        log.info("Dataset root: %s", dataset_root)
        n = rebuild_master_from_disk(dataset_root, motor_type=args.motor)
        log.info("=== Done: %d designs in master ===", n)
        return 0 if n > 0 else 2

    log.info("=== Multi-design dataset sweep: n=%d, seed=%d, start=%d, motor=%s ===",
             args.n, args.seed, args.start, args.motor)
    log.info("Log file: %s", log_path)
    log.info("Dataset root: %s", dataset_root)

    t0 = time.time()
    results = run_dataset(
        n_designs=args.n,
        master_seed=args.seed,
        dataset_root=dataset_root,
        n_phase3_speeds=args.phase3_speeds,
        n_phase3_torques=args.phase3_torques,
        start=args.start,
        motor_type=args.motor,
    )
    elapsed = time.time() - t0
    log.info("=== %d/%d designs succeeded in %.1f min ===",
             len(results), args.n, elapsed / 60.0)
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
