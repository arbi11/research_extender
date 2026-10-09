"""Stage 3: re-run FEMM at each (N*, T*) operating point and compute losses.

Workflow:
    1. Use the polynomial fits + TrajectorySolver to find an initial (i_d, i_q)
       guess that satisfies V/I limits and produces T close to the target.
    2. Run FEMM at that (i_d, i_q) to get the *true* T_em, lambda_d, lambda_q,
       and a peak B-field in the stator iron (for iron loss).
    3. Compute Cu loss (analytical), Fe loss (lumped Steinmetz), efficiency.

This stage IS slow (one FEMM solve per operating point), but it gives the
ground-truth data for the efficiency map. Skipping Phase-3 and trying to
estimate eta purely from polynomial fits would inherit the ~13 % torque-fit
error and the lossless d-q model assumption.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import Config
from .control_strategies import TrajectorySolver
from .dq_excitation import inverse_park, park_flux
from .losses import (
    IronLossParams,
    copper_loss_W,
    iron_loss_W,
    electrical_frequency_Hz,
    omega_mech_rad_s,
    efficiency,
    stator_iron_volume_m3,
)


log = logging.getLogger(__name__)


@dataclass
class Phase3Row:
    N_rpm: float
    T_target_Nm: float
    T_em_Nm: float          # FEMM measurement
    i_d_A: float
    i_q_A: float
    lambda_d_Wb: float
    lambda_q_Wb: float
    B_peak_iron_T: float
    P_cu_W: float
    P_fe_W: float
    P_mech_W: float
    eta: float
    regime: str             # MTPA / FW / MTPV / interior


def _stator_iron_b_peak(femm, geom) -> float:
    """Sample |B| on the stator back-iron at several azimuthal positions and
    return the peak. Cheap proxy for the (mass-averaged) peak B used in
    Steinmetz iron loss."""
    R_sample = (geom.R_stator_outer + (geom.R_stator_bore + geom.slot_depth)) / 2.0
    sample_angles_deg = np.linspace(0.0, 360.0, 24, endpoint=False)
    b_peak = 0.0
    for ang_deg in sample_angles_deg:
        x = R_sample * np.cos(np.radians(ang_deg))
        y = R_sample * np.sin(np.radians(ang_deg))
        bx, by = femm.mo_getb(x, y)
        b_mag = float(np.hypot(bx, by))
        if b_mag > b_peak:
            b_peak = b_mag
    return b_peak


def _solve_femm_at(femm, cfg: Config, i_d: float, i_q: float):
    """Set i_d, i_q in FEMM (at the configured electrical angle), solve, and
    return (T_em, lambda_d, lambda_q, B_peak_iron_T)."""
    theta_e = cfg.sweep.theta_elec_deg
    i_a, i_b, i_c = inverse_park(i_d, i_q, theta_e)
    femm.mi_modifycircprop("phase_A", 1, i_a)
    femm.mi_modifycircprop("phase_B", 1, i_b)
    femm.mi_modifycircprop("phase_C", 1, i_c)
    femm.mi_analyze(1)
    femm.mi_loadsolution()

    lam_a = femm.mo_getcircuitproperties("phase_A")[2]
    lam_b = femm.mo_getcircuitproperties("phase_B")[2]
    lam_c = femm.mo_getcircuitproperties("phase_C")[2]
    lam_d, lam_q = park_flux(lam_a, lam_b, lam_c, theta_e)

    femm.mo_groupselectblock(2)
    torque = femm.mo_blockintegral(22)
    femm.mo_clearblock()

    b_peak = _stator_iron_b_peak(femm, cfg.geom)
    return float(torque), float(lam_d), float(lam_q), b_peak


def build_target_grid(envelope_df: pd.DataFrame, n_speeds: int, n_torques: int) -> list[tuple[float, float]]:
    """Pick (N*, T*) targets evenly spaced under the operating envelope.

    For each of n_speeds speeds across [0, max_speed], use the Phase-2
    envelope to look up T_max(N), then distribute n_torques levels from
    a small positive value up to T_max(N).
    """
    if envelope_df.empty:
        return []
    feas = envelope_df[envelope_df["regime"] != "infeasible"].copy()
    if feas.empty:
        return []
    N_grid = np.linspace(feas["N_rpm"].min() + 100.0, feas["N_rpm"].max() - 100.0, n_speeds)
    targets: list[tuple[float, float]] = []
    for N in N_grid:
        T_env = float(np.interp(N, feas["N_rpm"].values, feas["T_em_Nm"].values))
        # Skip vanishingly small envelopes
        if T_env <= 0.5:
            continue
        T_levels = np.linspace(T_env * 0.05, T_env * 0.98, n_torques)
        for T in T_levels:
            targets.append((float(N), float(T)))
    return targets


def run_phase3_sweep(
    cfg: Config,
    solver: TrajectorySolver,
    envelope_df: pd.DataFrame,
    n_speeds: int = 8,
    n_torques: int = 6,
    iron_params: IronLossParams = IronLossParams(),
) -> pd.DataFrame:
    """Build geometry (caller already did), sweep targets, run FEMM at each.

    Assumes the FEMM model is already open with the IPM geometry meshed.
    """
    import femm  # noqa: WPS433

    targets = build_target_grid(envelope_df, n_speeds, n_torques)
    log.info("Phase 3 sweep: %d (N, T) target points", len(targets))

    V_iron = stator_iron_volume_m3(cfg.geom)
    log.info("Stator iron volume (approx): %.3e m^3", V_iron)

    rows: list[Phase3Row] = []
    for k, (N_rpm, T_target) in enumerate(targets, start=1):
        op = solver.solve_for_target_torque(N_rpm, T_target)
        if op is None:
            log.info("[%3d/%d] N=%.0f rpm, T*=%.2f Nm -> infeasible, skipping",
                     k, len(targets), N_rpm, T_target)
            continue

        t0 = time.time()
        T_em, lam_d, lam_q, B_peak = _solve_femm_at(femm, cfg, op.i_d_A, op.i_q_A)
        solve_s = time.time() - t0

        P_cu = copper_loss_W(op.i_d_A, op.i_q_A, cfg.limits.R_s_ohm)
        f_e = electrical_frequency_Hz(N_rpm, cfg.geom.num_pole_pairs)
        P_fe = iron_loss_W(B_peak, f_e, V_iron, iron_params)
        omega_m = omega_mech_rad_s(N_rpm)
        P_mech = T_em * omega_m
        eta = efficiency(T_em, omega_m, P_cu, P_fe)

        row = Phase3Row(
            N_rpm=N_rpm,
            T_target_Nm=T_target,
            T_em_Nm=T_em,
            i_d_A=op.i_d_A,
            i_q_A=op.i_q_A,
            lambda_d_Wb=lam_d,
            lambda_q_Wb=lam_q,
            B_peak_iron_T=B_peak,
            P_cu_W=P_cu,
            P_fe_W=P_fe,
            P_mech_W=P_mech,
            eta=eta,
            regime=op.regime,
        )
        rows.append(row)

        log.info(
            "[%3d/%d] N=%6.0f rpm  T*=%5.2f  T_em=%5.2f Nm  i=(%+5.1f, %+5.1f) A  "
            "B_pk=%.3f T  P_cu=%.1fW P_fe=%.1fW  eta=%5.3f  (%s, %.2fs)",
            k, len(targets), N_rpm, T_target, T_em,
            op.i_d_A, op.i_q_A, B_peak, P_cu, P_fe, eta, op.regime, solve_s,
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    if not df.empty:
        log.info("Phase 3 complete. eta range: [%.3f, %.3f]  (%d rows)",
                 df["eta"].min(), df["eta"].max(), len(df))
    return df
