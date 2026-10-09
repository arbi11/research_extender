"""Phase 4 driver: interpolate Phase-3 operating points into a dense
efficiency map and produce the Fig-0-style contour plot.

Run from native-Windows Python (no FEMM needed -- this is pure interpolation):

    python -m data_gen.run_phase4

Outputs:
    outputs/phase4_eta_grid.npz         dense (N, T, eta) grid for ML stage
    outputs/phase4_efficiency_map.png   contour plot styled like Fig 0
"""

from __future__ import annotations
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .stage4_efficiency_map import (
    interpolate_eta_map,
    save_grid_npz,
    plot_efficiency_map,
)


HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
OUT_DIR = HERE / "outputs"


def _setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"phase4_{ts}.log"
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
    log_path = _setup_logging()
    log = logging.getLogger("run_phase4")
    log.info("=== Phase 4: efficiency map ===")
    log.info("Log file: %s", log_path)

    points_csv = OUT_DIR / "phase3_eta_points.csv"
    env_csv = OUT_DIR / "phase2_trajectory.csv"
    if not points_csv.exists():
        log.error("Missing %s. Run Phase 3 first.", points_csv)
        return 2
    if not env_csv.exists():
        log.error("Missing %s. Run Phase 2 first.", env_csv)
        return 3

    points_df = pd.read_csv(points_csv)
    envelope_df = pd.read_csv(env_csv)
    log.info("Loaded %d phase-3 operating points and %d phase-2 envelope points",
             len(points_df), len(envelope_df))

    grid = interpolate_eta_map(points_df, envelope_df,
                               n_speed_grid=200, n_torque_grid=120)
    log.info("Interpolated grid: %d speeds x %d torques",
             grid["N_grid_rpm"].size, grid["T_grid_Nm"].size)
    finite = grid["eta_masked"][~np.isnan(grid["eta_masked"])]
    if finite.size:
        log.info("eta_masked: min=%.3f, max=%.3f, mean=%.3f",
                 float(finite.min()), float(finite.max()), float(finite.mean()))

    npz_path = OUT_DIR / "phase4_eta_grid.npz"
    save_grid_npz(npz_path, grid)
    log.info("Saved dense grid: %s", npz_path)

    png_path = OUT_DIR / "phase4_efficiency_map.png"
    plot_efficiency_map(grid, points_df, envelope_df, png_path)
    log.info("Wrote efficiency map: %s", png_path)
    log.info("=== Phase 4 done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
