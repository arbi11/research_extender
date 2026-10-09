"""Phase 3 driver: build geometry, sweep (N*, T*) operating points, run FEMM.

Outputs:
    outputs/phase3_eta_points.csv      one row per operating point
    outputs/phase3_eta_scatter.png     scatter of (N, T) coloured by eta

Run on a Windows box where pyFEMM + the FEMM .exe are installed:

    cd C:\\Users\\akhan\\...\\jupyter_book\\Chp3\\content
    python -m data_gen.run_phase3

Expects:
    outputs/phase1_fit_coeffs.npz       from Phase 1
    outputs/phase2_trajectory.csv       from Phase 2
"""

from __future__ import annotations
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config
from .flux_map_fit import PolyFit2D
from .geometry_ipm import build_ipm
from .control_strategies import TrajectorySolver
from .stage3_operating_points import run_phase3_sweep


HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
OUT_DIR = HERE / "outputs"


def _setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"phase3_{ts}.log"
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


def _load_fits() -> tuple[PolyFit2D, PolyFit2D, PolyFit2D]:
    npz_path = OUT_DIR / "phase1_fit_coeffs.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}. Run Phase 1 first.")
    z = np.load(npz_path)
    return (
        PolyFit2D(coeffs=z["lambda_d_coeffs"]),
        PolyFit2D(coeffs=z["lambda_q_coeffs"]),
        PolyFit2D(coeffs=z["torque_coeffs"]),
    )


def _load_envelope() -> pd.DataFrame:
    csv_path = OUT_DIR / "phase2_trajectory.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run Phase 2 first.")
    return pd.read_csv(csv_path)


def _scatter_plot(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["N_rpm"], df["T_em_Nm"], c=df["eta"] * 100.0,
                    cmap="viridis", s=60, edgecolor="k", linewidth=0.4,
                    vmin=df["eta"].min() * 100, vmax=df["eta"].max() * 100)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"Efficiency $\eta$ [%]")
    ax.set_xlabel("Speed N [RPM]")
    ax.set_ylabel(r"$T_{em}$ [N$\cdot$m]")
    ax.set_title("Phase 3 -- Operating points (FEMM + analytical losses)")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> int:
    log_path = _setup_logging()
    log = logging.getLogger("run_phase3")
    log.info("=== Phase 3: operating-point FEMM sweep ===")
    log.info("Log file: %s", log_path)

    cfg = Config()
    try:
        lam_d_fit, lam_q_fit, torque_fit = _load_fits()
        envelope_df = _load_envelope()
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    log.info("Loaded Phase 1 fits and Phase 2 envelope (%d points)", len(envelope_df))

    OUT_DIR.mkdir(exist_ok=True)
    fem_path = str(OUT_DIR / "ipm_phase3.fem")
    t_build = time.time()
    try:
        build_ipm(cfg, fem_path)
    except Exception:
        log.exception("Geometry build failed")
        return 3
    log.info("Geometry built in %.2f s", time.time() - t_build)

    # Mesh once; we only modify currents thereafter.
    import femm  # noqa: WPS433
    femm.mi_createmesh()

    solver = TrajectorySolver(
        lam_d_fit=lam_d_fit,
        lam_q_fit=lam_q_fit,
        torque_fit=torque_fit,
        pole_pairs=cfg.geom.num_pole_pairs,
        limits=cfg.limits,
    )

    try:
        df = run_phase3_sweep(cfg, solver, envelope_df, n_speeds=8, n_torques=6)
    except Exception:
        log.exception("Phase 3 sweep failed")
        return 4

    if df.empty:
        log.error("Phase 3 produced no operating points")
        return 5

    csv_path = OUT_DIR / "phase3_eta_points.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved %d rows to %s", len(df), csv_path)

    png_path = OUT_DIR / "phase3_eta_scatter.png"
    _scatter_plot(df, png_path)
    log.info("Wrote scatter plot: %s", png_path)

    # Summary stats per regime
    for r in df["regime"].unique():
        m = df["regime"] == r
        log.info("  %-10s: %3d pts, eta=[%5.3f, %5.3f], P_cu=[%6.1f, %6.1f]W, P_fe=[%6.1f, %6.1f]W",
                 r, m.sum(),
                 df.loc[m, "eta"].min(), df.loc[m, "eta"].max(),
                 df.loc[m, "P_cu_W"].min(), df.loc[m, "P_cu_W"].max(),
                 df.loc[m, "P_fe_W"].min(), df.loc[m, "P_fe_W"].max())

    log.info("=== Phase 3 done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
