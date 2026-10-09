"""Stage-1 characterization: sparse 9-point (i_d, i_q) FEA sweep.

For each (i_d, i_q) on the configured sparse grid:
    1. Inverse Park -> 3-phase peak currents (i_a, i_b, i_c).
    2. Write currents into FEMM circuits.
    3. Solve magnetostatic.
    4. Read phase flux linkages lambda_a, lambda_b, lambda_c from
       mo_getcircuitproperties.
    5. Park transform -> (lambda_d, lambda_q).
    6. Electromagnetic torque via mo_blockintegral(22) on the rotor group.

Returns a pandas.DataFrame with one row per sample point.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import Config, i_d_q_grid
from .dq_excitation import inverse_park, park_flux, validate_sum_zero


log = logging.getLogger(__name__)


@dataclass
class SamplePoint:
    i_d_A: float
    i_q_A: float
    i_a_A: float
    i_b_A: float
    i_c_A: float
    lambda_a_Wb: float
    lambda_b_Wb: float
    lambda_c_Wb: float
    lambda_d_Wb: float
    lambda_q_Wb: float
    torque_Nm: float
    solve_s: float


def _set_phase_currents(femm, i_a: float, i_b: float, i_c: float) -> None:
    femm.mi_modifycircprop("phase_A", 1, i_a)
    femm.mi_modifycircprop("phase_B", 1, i_b)
    femm.mi_modifycircprop("phase_C", 1, i_c)


def _solve_and_extract(femm, cfg: Config) -> tuple[float, float, float, float]:
    """Solve, then return (lambda_a, lambda_b, lambda_c, torque_Nm)."""
    femm.mi_analyze(1)        # 1 = hidden window
    femm.mi_loadsolution()

    # Phase flux linkages (index 2 of mo_getcircuitproperties output).
    lam_a = femm.mo_getcircuitproperties("phase_A")[2]
    lam_b = femm.mo_getcircuitproperties("phase_B")[2]
    lam_c = femm.mo_getcircuitproperties("phase_C")[2]

    # Torque on rotor group (group 2).
    femm.mo_groupselectblock(2)
    torque = femm.mo_blockintegral(22)
    femm.mo_clearblock()
    return float(lam_a), float(lam_b), float(lam_c), float(torque)


def run_sparse_sweep(cfg: Config) -> pd.DataFrame:
    """Run the sparse (i_d, i_q) sweep and return a tidy DataFrame.

    Assumes the FEMM model is already built and open (see geometry_ipm.build_ipm).
    """
    import femm  # noqa: WPS433

    grid = i_d_q_grid(cfg.sweep)
    theta_e = cfg.sweep.theta_elec_deg
    log.info("Starting sparse sweep: %d points, theta_e=%.1f deg", len(grid), theta_e)

    # Mesh once; we change currents only between solves, geometry is static.
    t0 = time.time()
    femm.mi_createmesh()
    log.info("Mesh created in %.2f s", time.time() - t0)

    samples: List[SamplePoint] = []
    for k, (i_d, i_q) in enumerate(grid, start=1):
        i_a, i_b, i_c = inverse_park(i_d, i_q, theta_e)
        validate_sum_zero(i_a, i_b, i_c)
        _set_phase_currents(femm, i_a, i_b, i_c)

        t_solve = time.time()
        lam_a, lam_b, lam_c, torque = _solve_and_extract(femm, cfg)
        solve_s = time.time() - t_solve

        lam_d, lam_q = park_flux(lam_a, lam_b, lam_c, theta_e)
        sp = SamplePoint(
            i_d_A=i_d, i_q_A=i_q,
            i_a_A=i_a, i_b_A=i_b, i_c_A=i_c,
            lambda_a_Wb=lam_a, lambda_b_Wb=lam_b, lambda_c_Wb=lam_c,
            lambda_d_Wb=lam_d, lambda_q_Wb=lam_q,
            torque_Nm=torque, solve_s=solve_s,
        )
        samples.append(sp)
        log.info(
            "[%2d/%d] i_d=%+6.1f i_q=%+6.1f  lam_d=%+8.4f Wb  lam_q=%+8.4f Wb  T=%+8.3f Nm  (%.2fs)",
            k, len(grid), i_d, i_q, lam_d, lam_q, torque, solve_s,
        )

    df = pd.DataFrame([s.__dict__ for s in samples])
    log.info(
        "Sweep complete.  lam_d in [%+.4f, %+.4f] Wb   lam_q in [%+.4f, %+.4f] Wb   T in [%+.3f, %+.3f] Nm",
        df.lambda_d_Wb.min(), df.lambda_d_Wb.max(),
        df.lambda_q_Wb.min(), df.lambda_q_Wb.max(),
        df.torque_Nm.min(), df.torque_Nm.max(),
    )
    return df
