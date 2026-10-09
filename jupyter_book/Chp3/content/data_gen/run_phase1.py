"""Phase 1 driver: characterize the IPM via a sparse (i_d, i_q) FEA sweep,
fit lambda_d / lambda_q with a 2nd-degree polynomial, and produce a Fig-5-style
3D-surface validation plot.

Run from native-Windows Python where pyFEMM and the FEMM .exe are installed:

    cd C:\\Users\\akhan\\code\\agam_dev\\graph_work\\research_extender-main\\the_code\\Chp3_PerformanceMapsPrediction
    python -m data_gen.run_phase1

Artefacts (under data_gen/outputs/):
    phase1_samples.csv       raw 9-point samples
    phase1_samples.h5        same data in HDF5
    phase1_fit_coeffs.npz    polynomial coefficients
    phase1_flux_surfaces.png 3D surface plot (vs Fig 5.PNG)

A timestamped log is also written under data_gen/logs/.
"""

from __future__ import annotations
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless; safe everywhere
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from . import config as cfg_mod
from .config import Config
from .geometry_ipm import build_ipm
from .stage1_characterize import run_sparse_sweep
from .flux_map_fit import fit_flux_maps, fit_quality, fit_torque


HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
OUT_DIR = HERE / "outputs"


def _setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"phase1_{ts}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Clear existing handlers so re-runs in the same process don't duplicate.
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


def _save_samples(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "phase1_samples.csv", index=False)
    try:
        df.to_hdf(OUT_DIR / "phase1_samples.h5", key="samples", mode="w")
    except (ImportError, ValueError) as exc:
        # pytables/tables may be missing; CSV is the canonical store.
        logging.getLogger(__name__).warning("HDF5 write skipped (%s)", exc)


def _save_fit_coeffs(lam_d_fit, lam_q_fit, torque_fit) -> None:
    np.savez(
        OUT_DIR / "phase1_fit_coeffs.npz",
        lambda_d_coeffs=lam_d_fit.coeffs,
        lambda_q_coeffs=lam_q_fit.coeffs,
        torque_coeffs=torque_fit.coeffs,
        column_order=np.array(["1", "d", "q", "d^2", "d*q", "q^2"], dtype="U8"),
    )


def _plot_flux_surfaces(df: pd.DataFrame, lam_d_fit, lam_q_fit, out_png: Path) -> None:
    """Reproduce the look of the_latex/Figures/Chp_RNN/Fig 5.PNG.

    Fig 5 plots axes in A_RMS and flux in mWb. We convert here at plot time.
    """
    sqrt2 = float(np.sqrt(2.0))

    # Dense evaluation grid (in peak amps for the fit).
    d_min, d_max = df["i_d_A"].min(), df["i_d_A"].max()
    q_min, q_max = df["i_q_A"].min(), df["i_q_A"].max()
    dd_peak = np.linspace(d_min, d_max, 41)
    qq_peak = np.linspace(q_min, q_max, 41)
    D_peak, Q_peak = np.meshgrid(dd_peak, qq_peak, indexing="xy")
    Ld = lam_d_fit(D_peak, Q_peak) * 1e3  # Wb -> mWb
    Lq = lam_q_fit(D_peak, Q_peak) * 1e3

    # Convert axes peak -> RMS for display (paper convention).
    D_rms = D_peak / sqrt2
    Q_rms = Q_peak / sqrt2
    d_pts_rms = df["i_d_A"].values / sqrt2
    q_pts_rms = df["i_q_A"].values / sqrt2
    ld_pts_mWb = df["lambda_d_Wb"].values * 1e3
    lq_pts_mWb = df["lambda_q_Wb"].values * 1e3

    fig = plt.figure(figsize=(13, 5.5))

    # --- lambda_d surface (left) ---
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(D_rms, Q_rms, Ld, cmap="viridis", alpha=0.85,
                     edgecolor="k", linewidth=0.15)
    ax1.scatter(d_pts_rms, q_pts_rms, ld_pts_mWb, color="black", s=30, depthshade=True)
    ax1.set_title(r"$\lambda_d$ [mWb]")
    ax1.set_xlabel(r"$I_d$ [A$_{RMS}$]")
    ax1.set_ylabel(r"$I_q$ [A$_{RMS}$]")
    ax1.set_zlabel(r"$\lambda_d$ [mWb]")
    ax1.view_init(elev=20, azim=-60)

    # --- lambda_q surface (right) ---
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(D_rms, Q_rms, Lq, cmap="viridis", alpha=0.85,
                     edgecolor="k", linewidth=0.15)
    ax2.scatter(d_pts_rms, q_pts_rms, lq_pts_mWb, color="black", s=30, depthshade=True)
    ax2.set_title(r"$\lambda_q$ [mWb]")
    ax2.set_xlabel(r"$I_d$ [A$_{RMS}$]")
    ax2.set_ylabel(r"$I_q$ [A$_{RMS}$]")
    ax2.set_zlabel(r"$\lambda_q$ [mWb]")
    ax2.view_init(elev=20, azim=-60)

    fig.suptitle("Phase 1 — Stage-1 flux-linkage maps (sparse FEA + 2nd-deg poly fit)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> int:
    log_path = _setup_logging()
    log = logging.getLogger("run_phase1")
    log.info("=== Phase 1: Stage-1 characterization ===")
    log.info("Log file: %s", log_path)

    cfg = Config()
    log.info("Geometry: %s", cfg.geom)
    log.info("Sweep grid:  i_d=%s  i_q=%s  theta_e=%.1f deg",
             cfg.sweep.i_d_peak_A, cfg.sweep.i_q_peak_A, cfg.sweep.theta_elec_deg)

    OUT_DIR.mkdir(exist_ok=True)
    fem_path = str(OUT_DIR / "ipm_phase1.fem")

    t_build = time.time()
    try:
        build_ipm(cfg, fem_path)
    except Exception:
        log.exception("Geometry build failed")
        return 2
    log.info("Geometry built in %.2f s", time.time() - t_build)

    try:
        df = run_sparse_sweep(cfg)
    except Exception:
        log.exception("Sweep failed")
        return 3

    _save_samples(df)
    log.info("Saved samples: outputs/phase1_samples.{csv,h5}")

    try:
        lam_d_fit, lam_q_fit = fit_flux_maps(df)
        torque_fit = fit_torque(df)
    except Exception:
        log.exception("Polynomial fit failed")
        return 4

    q_d = fit_quality(lam_d_fit, df["i_d_A"], df["i_q_A"], df["lambda_d_Wb"])
    q_q = fit_quality(lam_q_fit, df["i_d_A"], df["i_q_A"], df["lambda_q_Wb"])
    q_T = fit_quality(torque_fit, df["i_d_A"], df["i_q_A"], df["torque_Nm"])
    log.info("Fit quality lambda_d: %s", q_d)
    log.info("Fit quality lambda_q: %s", q_q)
    log.info("Fit quality torque:   %s", q_T)
    _save_fit_coeffs(lam_d_fit, lam_q_fit, torque_fit)

    out_png = OUT_DIR / "phase1_flux_surfaces.png"
    try:
        _plot_flux_surfaces(df, lam_d_fit, lam_q_fit, out_png)
    except Exception:
        log.exception("Plot failed")
        return 5
    log.info("Wrote plot: %s", out_png)
    log.info("=== Phase 1 done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
