"""Design-space sampling for the multi-design dataset sweep.

Defines the IPM geometry parameters that vary across the dataset, their ranges,
geometric feasibility checks, and a Latin-Hypercube sampler with rejection.

Per-design parameters varied (6):
    R_stator_outer       (mm)   stator outer radius
    R_stator_bore        (mm)   stator inner radius (bore)
    slot_depth           (mm)   radial slot depth
    magnet_length        (mm)   magnet radial extent
    magnet_width         (mm)   magnet tangential extent
    stack_length_mm      (mm)   axial depth

Kept fixed across designs (so the winding-pattern / pole-count match still holds):
    num_slots=12, num_poles=4, num_turns_per_slot=50, air gap=1 mm,
    rotor_offset_deg=135, flux_barrier_width=2 mm,
    R_rotor_outer = R_stator_bore - 1 (so air gap stays at 1 mm).
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import List, Tuple

import numpy as np
from scipy.stats import qmc

from .config import Config, FSCWGeometry, IPMGeometry


@dataclass(frozen=True)
class ParamRange:
    name: str
    low: float
    high: float


# Order of params here defines the order in the master dataset.
DESIGN_PARAM_RANGES: Tuple[ParamRange, ...] = (
    ParamRange("R_stator_outer", 95.0, 110.0),
    ParamRange("R_stator_bore",  55.0,  70.0),
    ParamRange("slot_depth",     20.0,  30.0),
    ParamRange("magnet_length",  16.0,  28.0),
    ParamRange("magnet_width",    6.0,  12.0),
    ParamRange("stack_length_mm", 25.0, 40.0),
)


def param_names() -> List[str]:
    return [pr.name for pr in DESIGN_PARAM_RANGES]


def geometry_from_params(params: np.ndarray, base: IPMGeometry | None = None) -> IPMGeometry:
    """Build an IPMGeometry from a parameter vector (length = len(DESIGN_PARAM_RANGES))."""
    base = base or IPMGeometry()
    if len(params) != len(DESIGN_PARAM_RANGES):
        raise ValueError(f"Expected {len(DESIGN_PARAM_RANGES)} params, got {len(params)}")
    overrides = {}
    for pr, v in zip(DESIGN_PARAM_RANGES, params):
        overrides[pr.name] = float(v)
    # Keep air gap fixed at 1 mm by setting R_rotor_outer = R_stator_bore - 1.
    overrides["R_rotor_outer"] = overrides["R_stator_bore"] - 1.0
    return replace(base, **overrides)


def is_feasible(geom: IPMGeometry) -> Tuple[bool, str]:
    """Geometric sanity checks. Returns (ok, reason_if_not)."""
    g = geom
    if g.R_stator_outer <= g.R_stator_bore + g.slot_depth:
        return False, "slot extends past stator outer radius (no back iron left)"
    if g.R_rotor_outer >= g.R_stator_bore:
        return False, "rotor crashes into stator bore"
    if g.R_rotor_outer <= g.R_rotor_inner + 5.0:
        return False, "rotor too thin radially"
    R_mc = (g.R_rotor_outer + g.R_rotor_inner) / 2.0
    if R_mc + g.magnet_length / 2.0 >= g.R_rotor_outer - 1.0:
        return False, "magnet outer face crashes into rotor OD (no bridge)"
    if R_mc - g.magnet_length / 2.0 <= g.R_rotor_inner + 1.0:
        return False, "magnet inner face crashes into rotor ID (no bridge)"
    # Pocket tangential width must fit within pole pitch with room left over.
    pocket_w = g.magnet_width + 2 * g.flux_barrier_width
    pocket_arc_deg = np.degrees(pocket_w / R_mc)
    pole_pitch_deg = 360.0 / g.num_poles
    if pocket_arc_deg > 0.7 * pole_pitch_deg:
        return False, (f"magnet pocket ({pocket_arc_deg:.1f} deg) "
                       f"exceeds 70% of pole pitch ({pole_pitch_deg:.1f} deg)")
    # Slot opening sanity (angular width must leave tooth between slots).
    slot_total_arc_deg = g.num_slots * g.slot_angle_deg
    if slot_total_arc_deg > 0.6 * 360.0:
        return False, "slots too wide relative to stator circumference"
    return True, "ok"


# -----------------------------------------------------------------------------
# FSCW (12-slot/10-pole surface-PM) design space
# -----------------------------------------------------------------------------

# Order defines the column order in the FSCW master dataset. Magnet thickness
# replaces magnet_length; magnet_width still tangential.
FSCW_DESIGN_PARAM_RANGES: Tuple[ParamRange, ...] = (
    ParamRange("R_stator_outer", 95.0, 110.0),
    ParamRange("R_stator_bore",  55.0,  70.0),
    ParamRange("slot_depth",     20.0,  30.0),
    ParamRange("magnet_thickness", 3.0,  6.0),
    ParamRange("magnet_width",    8.0, 16.0),
    ParamRange("stack_length_mm", 25.0, 40.0),
)


def fscw_param_names() -> List[str]:
    return [pr.name for pr in FSCW_DESIGN_PARAM_RANGES]


def geometry_from_params_fscw(
    params: np.ndarray, base: FSCWGeometry | None = None
) -> FSCWGeometry:
    base = base or FSCWGeometry()
    if len(params) != len(FSCW_DESIGN_PARAM_RANGES):
        raise ValueError(
            f"Expected {len(FSCW_DESIGN_PARAM_RANGES)} FSCW params, got {len(params)}"
        )
    overrides = {pr.name: float(v) for pr, v in zip(FSCW_DESIGN_PARAM_RANGES, params)}
    # Keep air gap fixed at 1 mm.
    overrides["R_rotor_outer"] = overrides["R_stator_bore"] - 1.0
    return replace(base, **overrides)


def is_feasible_fscw(geom: FSCWGeometry) -> Tuple[bool, str]:
    g = geom
    if g.R_stator_outer <= g.R_stator_bore + g.slot_depth:
        return False, "slot extends past stator outer radius (no back iron left)"
    if g.R_rotor_outer >= g.R_stator_bore:
        return False, "rotor crashes into stator bore"
    if g.R_rotor_outer <= g.R_rotor_inner + 5.0:
        return False, "rotor too thin radially"
    # Surface magnet must fit between rotor surface and rotor iron with 1mm steel margin.
    if g.magnet_thickness >= g.R_rotor_outer - g.R_rotor_inner - 2.0:
        return False, "magnet thickness leaves no steel under it"
    R_mag_c = g.R_rotor_outer - g.magnet_thickness / 2.0
    pole_pitch_mm = 2 * np.pi * R_mag_c / g.num_poles
    if g.magnet_width > 0.85 * pole_pitch_mm:
        return False, (f"magnet width {g.magnet_width:.2f} mm exceeds 85% of "
                       f"pole pitch {pole_pitch_mm:.2f} mm")
    if g.num_slots * g.slot_angle_deg > 0.6 * 360.0:
        return False, "slots too wide relative to stator circumference"
    return True, "ok"


def lhs_designs_fscw(n_designs: int, seed: int = 42, max_oversample: int = 8) -> np.ndarray:
    """LHS sampler over the FSCW design parameters with rejection."""
    n_params = len(FSCW_DESIGN_PARAM_RANGES)
    lows = np.array([pr.low for pr in FSCW_DESIGN_PARAM_RANGES])
    highs = np.array([pr.high for pr in FSCW_DESIGN_PARAM_RANGES])
    feasible: List[np.ndarray] = []
    n_request = n_designs
    while len(feasible) < n_designs and n_request <= max_oversample * n_designs:
        sampler = qmc.LatinHypercube(d=n_params, seed=seed + n_request)
        u = sampler.random(n=n_request)
        candidates = lows + u * (highs - lows)
        for row in candidates:
            geom = geometry_from_params_fscw(row)
            ok, _ = is_feasible_fscw(geom)
            if ok:
                feasible.append(row)
                if len(feasible) >= n_designs:
                    break
        n_request *= 2
    if len(feasible) < n_designs:
        raise RuntimeError(
            f"FSCW LHS: only found {len(feasible)} feasible designs out of "
            f"{max_oversample * n_designs} attempted."
        )
    return np.array(feasible[:n_designs])


# -----------------------------------------------------------------------------
# IPM LHS (original)
# -----------------------------------------------------------------------------

def lhs_designs(n_designs: int, seed: int = 42, max_oversample: int = 8) -> np.ndarray:
    """Generate n_designs FEASIBLE motor parameter vectors via LHS + rejection.

    Returns array of shape (n_designs, n_params). Raises if the sampler can't
    find enough feasible designs after max_oversample * n_designs attempts.
    """
    n_params = len(DESIGN_PARAM_RANGES)
    lows = np.array([pr.low for pr in DESIGN_PARAM_RANGES])
    highs = np.array([pr.high for pr in DESIGN_PARAM_RANGES])
    feasible: List[np.ndarray] = []

    n_request = n_designs
    while len(feasible) < n_designs and n_request <= max_oversample * n_designs:
        sampler = qmc.LatinHypercube(d=n_params, seed=seed + n_request)
        u = sampler.random(n=n_request)
        candidates = lows + u * (highs - lows)
        for row in candidates:
            geom = geometry_from_params(row)
            ok, _ = is_feasible(geom)
            if ok:
                feasible.append(row)
                if len(feasible) >= n_designs:
                    break
        n_request *= 2

    if len(feasible) < n_designs:
        raise RuntimeError(
            f"Could only find {len(feasible)} feasible designs out of "
            f"{max_oversample * n_designs} attempted. Loosen ranges or relax checks."
        )
    return np.array(feasible[:n_designs])
