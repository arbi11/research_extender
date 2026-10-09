"""Stage 4: interpolate the Phase-3 operating points onto a dense (N, T) grid
and build the final efficiency-map dataset.

Inputs:
    outputs/phase3_eta_points.csv   (N_rpm, T_em_Nm, eta, P_cu_W, P_fe_W, ...)
    outputs/phase2_trajectory.csv   (peak-torque envelope for masking)

Outputs:
    outputs/phase4_eta_grid.npz     dense interpolated map (N, T, eta, mask)
    outputs/phase4_efficiency_map.png   Fig-0-style contour plot
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


log = logging.getLogger(__name__)


def interpolate_eta_map(
    points_df: pd.DataFrame,
    envelope_df: pd.DataFrame,
    n_speed_grid: int = 200,
    n_torque_grid: int = 120,
    target_col: str = "eta",
) -> dict:
    """Interpolate the scatter onto a dense (N, T) grid.

    Returns a dict with:
        N_grid_rpm        shape (n_speed,)
        T_grid_Nm         shape (n_torque,)
        eta               shape (n_torque, n_speed)
        envelope_mask     shape (n_torque, n_speed)  True inside envelope
        eta_masked        eta with NaN outside envelope

    `target_col` selects which column of `points_df` to interpolate. Defaults to
    "eta" (efficiency). Pass "pf" to build a power-factor map with the same
    geometry. The output dict still uses the keys `eta` / `eta_masked` so
    downstream code stays unchanged; rename in the caller if needed.
    """
    if points_df.empty:
        raise ValueError("Phase 3 points are empty")

    N_min = float(points_df["N_rpm"].min())
    N_max = float(points_df["N_rpm"].max())
    T_min = 0.0
    T_max = float(points_df["T_em_Nm"].max())
    log.info("Target-map domain (%s): N in [%.0f, %.0f] rpm, T in [0, %.2f] Nm",
             target_col, N_min, N_max, T_max)

    N_grid = np.linspace(N_min, N_max, n_speed_grid)
    T_grid = np.linspace(T_min, T_max, n_torque_grid)
    N_mesh, T_mesh = np.meshgrid(N_grid, T_grid, indexing="xy")

    # Linear interpolation. Cubic can over/under-shoot near sharp eta gradients
    # at the envelope; linear is safer for this small scatter.
    pts = np.column_stack([points_df["N_rpm"].values, points_df["T_em_Nm"].values])
    values = points_df[target_col].values
    eta_lin = griddata(pts, values, (N_mesh, T_mesh), method="linear")
    # Fill any holes (just outside convex hull) with nearest-neighbour
    eta_nn = griddata(pts, values, (N_mesh, T_mesh), method="nearest")
    eta = np.where(np.isnan(eta_lin), eta_nn, eta_lin)

    # Build the operating-envelope mask from the Phase-2 trajectory.
    feas = envelope_df[envelope_df["regime"] != "infeasible"].copy()
    if not feas.empty:
        T_env_at_N = np.interp(
            N_mesh.ravel(),
            feas["N_rpm"].values,
            feas["T_em_Nm"].values,
            left=feas["T_em_Nm"].iloc[0],
            right=feas["T_em_Nm"].iloc[-1],
        ).reshape(N_mesh.shape)
        envelope_mask = T_mesh <= T_env_at_N
    else:
        envelope_mask = np.ones_like(N_mesh, dtype=bool)

    eta_masked = np.where(envelope_mask, eta, np.nan)
    return {
        "N_grid_rpm": N_grid,
        "T_grid_Nm": T_grid,
        "eta": eta,
        "envelope_mask": envelope_mask,
        "eta_masked": eta_masked,
        "envelope_T_at_N_grid": T_env_at_N if not feas.empty else None,
    }


def save_grid_npz(out_npz: Path, grid: dict) -> None:
    np.savez(
        out_npz,
        N_rpm=grid["N_grid_rpm"],
        T_Nm=grid["T_grid_Nm"],
        eta=grid["eta"],
        envelope_mask=grid["envelope_mask"],
        envelope_T_at_N=grid["envelope_T_at_N_grid"]
        if grid["envelope_T_at_N_grid"] is not None
        else np.array([]),
    )


def plot_efficiency_map(
    grid: dict,
    points_df: pd.DataFrame,
    envelope_df: pd.DataFrame,
    out_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = grid["N_grid_rpm"]
    T = grid["T_grid_Nm"]
    eta_pct = grid["eta_masked"] * 100.0

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Choose contour levels that emphasise the high-eta region (cf. Fig 0)
    finite_eta = eta_pct[~np.isnan(eta_pct)]
    if finite_eta.size == 0:
        log.warning("No finite eta values to plot")
        return
    lo = max(40.0, float(np.floor(finite_eta.min())))
    hi = float(np.ceil(finite_eta.max()))
    levels = np.concatenate([
        np.arange(lo, 90.0, 5.0),
        np.arange(90.0, hi + 0.5, 1.0),
    ])

    cf = ax.contourf(N, T, eta_pct, levels=levels, cmap="jet")
    cs = ax.contour(N, T, eta_pct, levels=levels, colors="k",
                    linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(r"Efficiency $\eta$ [%]")

    # Envelope overlay (peak torque-speed curve from Phase 2)
    feas = envelope_df[envelope_df["regime"] != "infeasible"]
    if not feas.empty:
        ax.plot(feas["N_rpm"], feas["T_em_Nm"], "k-", linewidth=2.0,
                label="Peak Torque/Power curve")

    # Scatter the actual FEMM operating points used for the interpolation
    ax.scatter(points_df["N_rpm"], points_df["T_em_Nm"],
               c="white", edgecolor="k", s=22, linewidth=0.5,
               label="FEMM samples", zorder=3)

    ax.set_xlabel("Speed [RPM]")
    ax.set_ylabel(r"Torque [N$\cdot$m]")
    ax.set_title("Phase 4 -- Efficiency map  η(N, T)")
    ax.set_xlim(N.min(), N.max())
    ax.set_ylim(0, T.max() * 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
