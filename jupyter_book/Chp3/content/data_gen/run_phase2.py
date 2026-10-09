"""Phase 2 driver: build the torque-speed envelope from Phase 1 flux fits.

Loads outputs/phase1_fit_coeffs.npz (written by run_phase1.py), constructs
the TrajectorySolver, sweeps speed from 0 to max_speed_rpm, and writes:

    outputs/phase2_trajectory.csv     (one row per speed)
    outputs/phase2_envelope.png       (T-N envelope, current trajectory, U-N)

This stage uses NO FEMM. It runs entirely on the polynomial fits, so it is
fast (sub-second on a 401x401 grid).
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
from .control_strategies import TrajectorySolver, OperatingPoint


HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
OUT_DIR = HERE / "outputs"


def _setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"phase2_{ts}.log"
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
        raise FileNotFoundError(
            f"Missing {npz_path}. Run Phase 1 first: python -m data_gen.run_phase1"
        )
    z = np.load(npz_path)
    if "torque_coeffs" not in z.files:
        raise KeyError(
            "phase1_fit_coeffs.npz is missing 'torque_coeffs'. "
            "Re-run Phase 1 after the torque-fit upgrade."
        )
    return (
        PolyFit2D(coeffs=z["lambda_d_coeffs"]),
        PolyFit2D(coeffs=z["lambda_q_coeffs"]),
        PolyFit2D(coeffs=z["torque_coeffs"]),
    )


def _trajectory_to_df(traj: list[OperatingPoint]) -> pd.DataFrame:
    return pd.DataFrame([op.__dict__ for op in traj])


def _plot_envelope(df: pd.DataFrame, cfg: Config, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    feasible = df["regime"] != "infeasible"
    regimes = df["regime"].unique()
    colors = {"MTPA": "tab:blue", "FW": "tab:orange", "MTPV": "tab:green",
              "interior": "tab:gray", "infeasible": "tab:red"}

    # --- (1) Torque-speed envelope ---
    ax = axes[0]
    for r in regimes:
        m = (df["regime"] == r) & feasible
        ax.plot(df.loc[m, "N_rpm"], df.loc[m, "T_em_Nm"], "o-",
                color=colors.get(r, "k"), label=r, markersize=4)
    ax.set_xlabel("Speed N [RPM]")
    ax.set_ylabel(r"$T_{em}^*$ [N$\cdot$m]")
    ax.set_title("Torque-Speed Envelope")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best", fontsize=9)

    # --- (2) Optimal current trajectory in (i_d, i_q) ---
    ax = axes[1]
    I_max = cfg.limits.I_max_peak_A
    circle_phi = np.linspace(0, 2 * np.pi, 200)
    ax.plot(I_max * np.cos(circle_phi), I_max * np.sin(circle_phi),
            "k--", alpha=0.4, label=f"|I|={I_max}A")
    for r in regimes:
        m = (df["regime"] == r) & feasible
        ax.plot(df.loc[m, "i_d_A"], df.loc[m, "i_q_A"], "o-",
                color=colors.get(r, "k"), label=r, markersize=4)
    ax.set_xlabel(r"$i_d$ [A peak]")
    ax.set_ylabel(r"$i_q$ [A peak]")
    ax.set_title("Optimal current trajectory")
    ax.grid(True, alpha=0.4)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=9)

    # --- (3) |U| vs speed (showing V-limit) ---
    ax = axes[2]
    ax.plot(df["N_rpm"], df["U_mag_V"], "o-", color="tab:purple", markersize=4)
    ax.axhline(cfg.limits.V_max_peak_V, color="r", linestyle="--",
               label=f"V_max = {cfg.limits.V_max_peak_V:.0f} V")
    ax.set_xlabel("Speed N [RPM]")
    ax.set_ylabel(r"$|U|$ [V peak]")
    ax.set_title("Phase voltage magnitude")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best", fontsize=9)

    fig.suptitle("Phase 2 -- Optimal operating-point trajectory "
                 f"(p={cfg.geom.num_pole_pairs}, lambda_PM~{cfg.limits.V_max_peak_V*0+10} mWb)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> int:
    log_path = _setup_logging()
    log = logging.getLogger("run_phase2")
    log.info("=== Phase 2: control-strategy trajectory ===")
    log.info("Log file: %s", log_path)

    cfg = Config()
    log.info("Limits: I_max=%.1f A peak, V_max=%.1f V peak, R_s=%.3f ohm, N_max=%.0f rpm",
             cfg.limits.I_max_peak_A, cfg.limits.V_max_peak_V,
             cfg.limits.R_s_ohm, cfg.limits.max_speed_rpm)
    log.info("Machine: p=%d pole pairs", cfg.geom.num_pole_pairs)

    try:
        lam_d_fit, lam_q_fit, torque_fit = _load_fits()
    except (FileNotFoundError, KeyError) as exc:
        log.error("%s", exc)
        return 2
    log.info("Loaded Phase 1 fits (lambda_d, lambda_q, torque) from outputs/phase1_fit_coeffs.npz")

    t0 = time.time()
    solver = TrajectorySolver(
        lam_d_fit=lam_d_fit,
        lam_q_fit=lam_q_fit,
        torque_fit=torque_fit,
        pole_pairs=cfg.geom.num_pole_pairs,
        limits=cfg.limits,
    )
    log.info("Solver grid built in %.2f s", time.time() - t0)

    traj = solver.sweep_speed(n_speeds=80)
    log.info("Computed %d operating points across speed sweep", len(traj))

    df = _trajectory_to_df(traj)
    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / "phase2_trajectory.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved trajectory CSV: %s", csv_path)

    # Quick summary printed to log.
    feas = df[df["regime"] != "infeasible"]
    if not feas.empty:
        log.info("T_em range:   [%+.2f, %+.2f] Nm",
                 feas["T_em_Nm"].min(), feas["T_em_Nm"].max())
        log.info("N range:      [%.0f, %.0f] rpm",
                 feas["N_rpm"].min(), feas["N_rpm"].max())
        for r in feas["regime"].unique():
            m = feas["regime"] == r
            log.info("  %-10s : %d points (N=[%.0f, %.0f] rpm)",
                     r, m.sum(), feas.loc[m, "N_rpm"].min(), feas.loc[m, "N_rpm"].max())

    png_path = OUT_DIR / "phase2_envelope.png"
    try:
        _plot_envelope(df, cfg, png_path)
        log.info("Wrote plot: %s", png_path)
    except Exception:
        log.exception("Plot failed")
        return 3

    log.info("=== Phase 2 done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
