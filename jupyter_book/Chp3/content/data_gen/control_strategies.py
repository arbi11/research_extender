"""Stage-2: optimal d-q current trajectory under current + voltage limits.

For a given mechanical speed N_rpm, the solver finds (i_d*, i_q*) that
MAXIMIZES electromagnetic torque subject to:

    i_d^2 + i_q^2 <= I_max^2                                 (current circle)
    (R_s*i_d - omega_e*lambda_q)^2 + (R_s*i_q + omega_e*lambda_d)^2
        <= V_max^2                                          (voltage ellipse)

with omega_e = p * 2*pi * N_rpm / 60.

This single formulation reproduces MTPA (only the current limit binds, at low
speeds), field weakening (both limits bind, intermediate speeds), and MTPV
(voltage limit dominates, very high speeds) without needing three separate
solvers.

The flux maps are evaluated through the polynomial fits from Phase 1, so each
function call is cheap. We discretize (i_d, i_q) on a dense grid in the
operationally relevant quadrant (i_d <= 0, i_q >= 0) and pick the maximum-T
point that satisfies both constraints.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import MachineLimits
from .flux_map_fit import PolyFit2D


log = logging.getLogger(__name__)


@dataclass
class OperatingPoint:
    N_rpm: float
    i_d_A: float
    i_q_A: float
    T_em_Nm: float
    lambda_d_Wb: float
    lambda_q_Wb: float
    u_d_V: float
    u_q_V: float
    I_mag_A: float
    U_mag_V: float
    regime: str       # "MTPA" / "FW" / "MTPV" / "infeasible"


@dataclass
class TrajectorySolver:
    lam_d_fit: PolyFit2D
    lam_q_fit: PolyFit2D
    torque_fit: PolyFit2D    # T(i_d, i_q) fitted from FEMM Maxwell-stress samples
    pole_pairs: int
    limits: MachineLimits
    n_grid: int = 401      # grid points per axis in the (i_d, i_q) search

    def __post_init__(self):
        I_max = self.limits.I_max_peak_A
        # Operationally relevant quadrant for an IPM motoring: i_d in [-I_max, 0],
        # i_q in [0, I_max]. Extending slightly past zero on i_d / past 0 on i_q
        # would matter for generating; ignore for now.
        i_d = np.linspace(-I_max, 0.0, self.n_grid)
        i_q = np.linspace(0.0, I_max, self.n_grid)
        I_d, I_q = np.meshgrid(i_d, i_q, indexing="ij")
        self._I_d = I_d
        self._I_q = I_q

        # Pre-compute the flux maps and torque on the grid. Use the
        # FEMM-fitted torque (NOT the d-q analytical formula) since our
        # non-sinusoidal rectangular magnets break the fundamental-only model.
        self._L_d = self.lam_d_fit(I_d, I_q)
        self._L_q = self.lam_q_fit(I_d, I_q)
        self._T = self.torque_fit(I_d, I_q)
        self._I_mag = np.hypot(I_d, I_q)
        self._I_feasible = self._I_mag <= I_max
        log.info("TrajectorySolver: %dx%d grid, I_max=%.1f A, V_max=%.1f V, p=%d",
                 self.n_grid, self.n_grid, I_max,
                 self.limits.V_max_peak_V, self.pole_pairs)
        log.info("  Grid torque range (from FEMM-fitted T):  T in [%+.2f, %+.2f] Nm",
                 float(self._T.min()), float(self._T.max()))

    # -----------------------------------------------------------------
    def voltage_components(self, N_rpm: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (u_d, u_q) on the precomputed (i_d, i_q) grid for given N."""
        omega_e = self.pole_pairs * 2 * np.pi * N_rpm / 60.0
        R_s = self.limits.R_s_ohm
        u_d = R_s * self._I_d - omega_e * self._L_q
        u_q = R_s * self._I_q + omega_e * self._L_d
        return u_d, u_q

    def solve_speed(self, N_rpm: float) -> OperatingPoint:
        """Find the max-torque (i_d, i_q) satisfying both limits at speed N_rpm."""
        u_d, u_q = self.voltage_components(N_rpm)
        U_mag = np.hypot(u_d, u_q)
        V_feasible = U_mag <= self.limits.V_max_peak_V
        feasible = V_feasible & self._I_feasible

        if not feasible.any():
            return OperatingPoint(
                N_rpm=N_rpm, i_d_A=np.nan, i_q_A=np.nan, T_em_Nm=0.0,
                lambda_d_Wb=np.nan, lambda_q_Wb=np.nan,
                u_d_V=np.nan, u_q_V=np.nan,
                I_mag_A=np.nan, U_mag_V=np.nan, regime="infeasible",
            )

        T_search = np.where(feasible, self._T, -np.inf)
        idx = np.unravel_index(int(np.argmax(T_search)), T_search.shape)

        # Classify regime: which constraint is active at the optimum?
        on_I = abs(self._I_mag[idx] - self.limits.I_max_peak_A) < 0.5  # within 0.5 A of I_max
        on_V = abs(U_mag[idx] - self.limits.V_max_peak_V) < 0.5         # within 0.5 V of V_max
        if on_I and not on_V:
            regime = "MTPA"
        elif on_I and on_V:
            regime = "FW"
        elif on_V and not on_I:
            regime = "MTPV"
        else:
            regime = "interior"

        return OperatingPoint(
            N_rpm=N_rpm,
            i_d_A=float(self._I_d[idx]),
            i_q_A=float(self._I_q[idx]),
            T_em_Nm=float(self._T[idx]),
            lambda_d_Wb=float(self._L_d[idx]),
            lambda_q_Wb=float(self._L_q[idx]),
            u_d_V=float(u_d[idx]),
            u_q_V=float(u_q[idx]),
            I_mag_A=float(self._I_mag[idx]),
            U_mag_V=float(U_mag[idx]),
            regime=regime,
        )

    def sweep_speed(self, n_speeds: int = 80) -> list[OperatingPoint]:
        """Return a list of operating points across 0 -> max_speed_rpm."""
        speeds = np.linspace(0.0, self.limits.max_speed_rpm, n_speeds)
        return [self.solve_speed(float(N)) for N in speeds]

    # -----------------------------------------------------------------
    def solve_for_target_torque(
        self,
        N_rpm: float,
        T_target_Nm: float,
        T_tolerance_Nm: float = 0.3,
    ) -> Optional[OperatingPoint]:
        """Find (i_d, i_q) producing T_target_Nm at speed N_rpm with minimum |I|.

        Used by Stage 3 to populate the (N, T) efficiency map: for each
        requested torque level, returns the lowest-current operating point
        that:
            - satisfies the current limit (|I| <= I_max)
            - satisfies the voltage limit at this speed
            - delivers torque within +/- T_tolerance_Nm of T_target_Nm

        Returns None if no point in the searchable quadrant satisfies all of
        the above (e.g. T_target exceeds the achievable envelope at this speed).
        """
        u_d, u_q = self.voltage_components(N_rpm)
        U_mag = np.hypot(u_d, u_q)
        feasible = (
            self._I_feasible
            & (U_mag <= self.limits.V_max_peak_V)
            & (np.abs(self._T - T_target_Nm) <= T_tolerance_Nm)
        )
        if not feasible.any():
            return None

        # Among feasible points, minimise current magnitude (proxy for Cu loss).
        I_search = np.where(feasible, self._I_mag, np.inf)
        idx = np.unravel_index(int(np.argmin(I_search)), I_search.shape)

        on_I = abs(self._I_mag[idx] - self.limits.I_max_peak_A) < 0.5
        on_V = abs(U_mag[idx] - self.limits.V_max_peak_V) < 0.5
        if on_I and not on_V:
            regime = "MTPA"
        elif on_I and on_V:
            regime = "FW"
        elif on_V and not on_I:
            regime = "MTPV"
        else:
            regime = "interior"

        return OperatingPoint(
            N_rpm=N_rpm,
            i_d_A=float(self._I_d[idx]),
            i_q_A=float(self._I_q[idx]),
            T_em_Nm=float(self._T[idx]),
            lambda_d_Wb=float(self._L_d[idx]),
            lambda_q_Wb=float(self._L_q[idx]),
            u_d_V=float(u_d[idx]),
            u_q_V=float(u_q[idx]),
            I_mag_A=float(self._I_mag[idx]),
            U_mag_V=float(U_mag[idx]),
            regime=regime,
        )
