"""12s/10p FSCW PM motor — DOUBLE-LAYER winding variant (v2).  EXPERIMENTAL.

::: STATUS (2026-05-26) :::
The slot-subdivision INFRASTRUCTURE works correctly: each slot is split
azimuthally by a radial divider line into two sub-regions (left/right), and a
distinct block label is placed in each sub-region. Mesh and solve succeed.

BUT the candidate tooth-coil pattern hard-coded in `_FSCW_V2_TOOTH_PHASES`
and `_FSCW_V2_TOOTH_POLARITIES` produces a winding whose phase-axis spacing
at the rotor's working harmonic (p=5) is ~30 degrees electrical rather than
the 120 degrees required for balanced 3-phase operation. Empirical result:
loaded torque collapsed to ~0.09 Nm at 50 A peak (vs v1's ~3.9 Nm with the
single-layer AACCBB pattern + second-half polarity flip).

For the FSCW dataset sweep, USE v1 (`geometry_fscw.py`) — it produces
physically meaningful torque and design-parameter response. v2 should be
treated as scaffolding for a future iteration that consults the EMETOR
atlas (or implements an empirical winding-factor optimiser) to find the
correct tooth-coil layout for textbook kw=0.933 operation.

Differences from `geometry_fscw.py` (v1):
    - Each slot is split AZIMUTHALLY by a radial divider line into two
      sub-regions (left/right), each carrying a different tooth coil's side.
    - Winding pattern is a per-TOOTH layout (one coil per tooth);
      each slot's two halves are filled from the coils of the two adjacent
      teeth.

v1 (rotor + winding) is kept untouched so the LHS dataset sweep that
uses it can keep running. Only `build_fscw_v2()` and its
`_define_circuits_and_coils_double_layer()` differ from v1.

Public entry point:
    build_fscw_v2(cfg, fem_path) -> None
"""

from __future__ import annotations
import logging
from typing import List, Tuple

import numpy as np

from .config import Config
from .geometry_ipm import (
    _draw_circle_arcs,
    _draw_slots,
    _open_femm,
    _define_materials,
    _add_airbox_and_bc,
)


log = logging.getLogger(__name__)

_LabelXY = Tuple[float, float]


# Discretisation per arc segment when calling mi_drawarc. The mesher subdivides
# further if it needs to; this is the max-segments hint.
_ARC_SEGMENTS = 5


# Double-layer tooth-coil pattern (one entry per TOOTH, not per slot).
# Tooth k sits between slot k and slot k+1 (modulo num_slots).
# This is the standard "ABCABCABC..." layout for 12s/10p concentrated windings:
# each tooth carries one coil from a distinct phase, with first-half-vs-second-
# half polarity flip (so the two pole-pair patches add at p=5 instead of cancelling).
#
# This is a CANDIDATE pattern to test empirically against v1. If a different
# tooth-coil arrangement turns out to give cleaner balanced 3-phase MMF, change
# this constant and rerun the debug harness.
_FSCW_V2_TOOTH_PHASES: Tuple[str, ...] = (
    "A", "B", "C", "A", "B", "C",
    "A", "B", "C", "A", "B", "C",
)
_FSCW_V2_TOOTH_POLARITIES: Tuple[int, ...] = (
    +1, -1, +1, -1, +1, -1,
    -1, +1, -1, +1, -1, +1,   # second-half polarities flipped
)


def _magnet_half_angle_rad(cfg: Config) -> float:
    """Half-angular-extent of one magnet sector (radians).

    `magnet_width` is interpreted as the arc length at the sector mid-radius
    (R_rotor_outer − magnet_thickness/2). For thin magnets this is
    indistinguishable from the chord length.
    """
    g = cfg.geom
    R_mid = g.R_rotor_outer - g.magnet_thickness / 2.0
    return (g.magnet_width / 2.0) / R_mid


def _draw_rotor_surface(femm, cfg: Config) -> None:
    """Draw the rotor outer boundary as alternating *magnet sectors* and
    *inter-magnet steel arcs*.

    Surface-mount PM topology: each magnet's outer arc IS part of the rotor
    outer boundary. Between magnets, a steel arc closes the boundary at the
    same R_rotor_outer. Magnet outer arcs and steel arcs share endpoints
    exactly (computed from the same angles), so the mesher sees a clean
    closed polyline of arcs with no crossings and no slivers.

    Each magnet is a true sector defined by:
        - outer arc at R_rotor_outer (the rotor's outer face there)
        - inner arc at R_rotor_outer − magnet_thickness
        - two radial line segments connecting outer to inner arcs

    The rotor outer arc is therefore NOT drawn separately by
    `_draw_circle_arcs(R_rotor_outer)` — it is composed entirely of the
    arcs drawn here.
    """
    g = cfg.geom
    N = g.num_poles
    R_outer = g.R_rotor_outer
    R_inner_mag = R_outer - g.magnet_thickness  # magnet inner face = rotor steel boundary

    if R_inner_mag < g.R_rotor_inner + 1.0:
        raise ValueError(
            f"Magnet inner face {R_inner_mag:.2f} mm too close to rotor ID "
            f"{g.R_rotor_inner:.2f} mm (need 1 mm clearance)"
        )

    half_alpha_rad = _magnet_half_angle_rad(cfg)
    half_alpha_deg = np.degrees(half_alpha_rad)
    pole_pitch_deg = 360.0 / N
    if 2.0 * half_alpha_deg > 0.95 * pole_pitch_deg:
        raise ValueError(
            f"Magnet angular span {2*half_alpha_deg:.2f} deg exceeds 95% of "
            f"pole pitch {pole_pitch_deg:.2f} deg — magnets would touch."
        )

    log.debug(
        "Sector magnets: %d poles, span=%.3f deg each, pole_pitch=%.3f deg, "
        "magnet_thickness=%.2f mm",
        N, 2 * half_alpha_deg, pole_pitch_deg, g.magnet_thickness,
    )

    def point(R: float, angle_rad: float) -> Tuple[float, float]:
        return R * np.cos(angle_rad), R * np.sin(angle_rad)

    for p in range(N):
        theta_pole = np.radians(p * pole_pitch_deg + g.rotor_offset_deg)
        a_mag_start = theta_pole - half_alpha_rad
        a_mag_end = theta_pole + half_alpha_rad

        # Magnet outer arc (= rotor outer surface across this magnet).
        x_o_s, y_o_s = point(R_outer, a_mag_start)
        x_o_e, y_o_e = point(R_outer, a_mag_end)
        femm.mi_drawarc(x_o_s, y_o_s, x_o_e, y_o_e,
                        2 * half_alpha_deg, _ARC_SEGMENTS)

        # Magnet inner arc (= rotor-steel boundary under this magnet).
        x_i_s, y_i_s = point(R_inner_mag, a_mag_start)
        x_i_e, y_i_e = point(R_inner_mag, a_mag_end)
        femm.mi_drawarc(x_i_s, y_i_s, x_i_e, y_i_e,
                        2 * half_alpha_deg, _ARC_SEGMENTS)

        # Two radial side lines closing the magnet sector.
        femm.mi_drawline(x_o_s, y_o_s, x_i_s, y_i_s)
        femm.mi_drawline(x_o_e, y_o_e, x_i_e, y_i_e)

        # Inter-magnet steel arc, from this magnet's end to the next one's start.
        theta_pole_next = np.radians((p + 1) * pole_pitch_deg + g.rotor_offset_deg)
        a_gap_end = theta_pole_next - half_alpha_rad
        gap_span_deg = np.degrees(a_gap_end - a_mag_end)
        if gap_span_deg <= 1e-6:
            raise ValueError(
                f"Non-positive steel-arc span between poles {p} and {p+1}: "
                f"{gap_span_deg:.4f} deg"
            )
        x_g_s, y_g_s = x_o_e, y_o_e  # exactly the magnet end-point we just drew
        x_g_e, y_g_e = point(R_outer, a_gap_end)
        femm.mi_drawarc(x_g_s, y_g_s, x_g_e, y_g_e,
                        gap_span_deg, _ARC_SEGMENTS)


def _place_labels(femm, cfg: Config) -> List[_LabelXY]:
    """Place material block labels for FSCW: stator iron, air gap, rotor iron,
    rotor bore air, and one magnet per pole (alternating radial magnetisation).
    """
    g = cfg.geom
    labels: List[_LabelXY] = []

    # 1. Stator back iron - between slots
    r = (g.R_stator_outer + (g.R_stator_bore + g.slot_depth)) / 2.0
    a = np.radians(360.0 / g.num_slots / 2.0)
    sx, sy = r * np.cos(a), r * np.sin(a)
    femm.mi_addblocklabel(sx, sy)
    femm.mi_selectlabel(sx, sy)
    femm.mi_setblockprop(cfg.mat.stator_steel, 1, 0, "<None>", 0, 1, 0)
    femm.mi_clearselected()
    labels.append((sx, sy))

    # 2. Air gap (refined mesh)
    rg = (g.R_stator_bore + g.R_rotor_outer) / 2.0
    femm.mi_addblocklabel(rg, 0.0)
    femm.mi_selectlabel(rg, 0.0)
    femm.mi_setblockprop("Air", 1, cfg.solver.air_gap_mesh_mm, "<None>", 0, 0, 0)
    femm.mi_clearselected()
    labels.append((rg, 0.0))

    # 3. Rotor back iron - well inside the rotor below the magnet ring.
    # Magnet centre radius (mid-thickness of the sector); used here for labels only.
    R_mag_c = g.R_rotor_outer - g.magnet_thickness / 2.0
    R_iron_label = (g.R_rotor_outer - g.magnet_thickness + g.R_rotor_inner) / 2.0
    # Angular position midway between two magnets so the label is in steel.
    half_pole_deg = (360.0 / g.num_poles) / 2.0
    a_iron = np.radians(half_pole_deg + g.rotor_offset_deg)
    rbx, rby = R_iron_label * np.cos(a_iron), R_iron_label * np.sin(a_iron)
    femm.mi_addblocklabel(rbx, rby)
    femm.mi_selectlabel(rbx, rby)
    femm.mi_setblockprop(cfg.mat.rotor_steel, 1, 0, "<None>", 0, 2, 0)
    femm.mi_clearselected()
    labels.append((rbx, rby))

    # 4. Rotor inner bore (air at origin, group 2 so it moves with the rotor)
    femm.mi_addblocklabel(0.0, 0.0)
    femm.mi_selectlabel(0.0, 0.0)
    femm.mi_setblockprop("Air", 1, 0, "<None>", 0, 2, 0)
    femm.mi_clearselected()
    labels.append((0.0, 0.0))

    # 5. Magnet labels: alternating radial magnetisation (group 2).
    # Place the label on the radial centreline at the sector mid-radius.
    # The sector is bounded by the magnet outer/inner arcs, so the radial
    # midline through the pole angle always falls inside the magnet.
    for p in range(g.num_poles):
        theta_deg = p * (360.0 / g.num_poles) + g.rotor_offset_deg
        theta = np.radians(theta_deg)
        cx, cy = R_mag_c * np.cos(theta), R_mag_c * np.sin(theta)
        mag_dir = theta_deg if p % 2 == 0 else theta_deg + 180.0
        femm.mi_addblocklabel(cx, cy)
        femm.mi_selectlabel(cx, cy)
        femm.mi_setblockprop(cfg.mat.magnet_name, 1, 0, "<None>", mag_dir, 2, 0)
        femm.mi_clearselected()
        labels.append((cx, cy))

    return labels


def _define_circuits_and_coils_double_layer(femm, cfg: Config) -> None:
    """True double-layer FSCW winding.

    Each slot is split AZIMUTHALLY by a radial line at the slot's centre
    azimuth into two regions:
        - LEFT  (azimuthally lower half, between (theta - slot_angle/2) and theta)
                carries the "return" side of tooth (k-1)'s coil.
        - RIGHT (azimuthally upper half, between theta and (theta + slot_angle/2))
                carries the "go" side of tooth k's coil.

    This matches the physical layout of tooth-wound concentrated coils: each
    coil spans one tooth, with its two sides in the two slots flanking that
    tooth. Slot k therefore sees ONE side of each adjacent tooth's coil — and
    those two coil sides come from different phases for the standard
    12s/10p ABC tooth pattern (see `_FSCW_V2_TOOTH_PHASES`).

    Convention:
        Right half of slot k = tooth k's coil "go" side
            -> phase = _FSCW_V2_TOOTH_PHASES[k]
            -> turns = +_FSCW_V2_TOOTH_POLARITIES[k] * turns_per_slot
        Left half of slot k  = tooth (k-1)'s coil "return" side
            -> phase = _FSCW_V2_TOOTH_PHASES[k-1]
            -> turns = -_FSCW_V2_TOOTH_POLARITIES[k-1] * turns_per_slot
    Tooth index (-1) wraps to tooth (num_slots - 1).
    """
    femm.mi_addcircprop("phase_A", 0.0, 1)
    femm.mi_addcircprop("phase_B", 0.0, 1)
    femm.mi_addcircprop("phase_C", 0.0, 1)

    g = cfg.geom
    R_bore = g.R_stator_bore
    R_bot = g.R_stator_bore + g.slot_depth
    R_lbl = 0.5 * (R_bore + R_bot)  # label sits at the slot mid-depth radially

    half_slot_angle = np.radians(g.slot_angle_deg) / 2.0
    # Place each label one-quarter slot width to each side of the divider so
    # it lands cleanly inside its sub-region (not on the divider line itself).
    quarter_slot_angle = half_slot_angle / 2.0

    n_teeth = g.num_slots  # one coil per tooth for 12s/10p concentrated winding
    if len(_FSCW_V2_TOOTH_PHASES) != n_teeth:
        raise ValueError(
            f"_FSCW_V2_TOOTH_PHASES has {len(_FSCW_V2_TOOTH_PHASES)} entries "
            f"but num_slots={n_teeth}; double-layer winding pattern is "
            "currently hardcoded for 12s/10p only."
        )

    # 1. Draw the radial divider line at the slot centre azimuth, from the
    #    bore arc to the slot-bottom arc. Endpoints land EXACTLY on the
    #    existing slot-boundary arcs so FEMM node-snapping fuses them and
    #    the slot region is genuinely subdivided into left/right halves.
    for k in range(g.num_slots):
        theta = 2 * np.pi * k / g.num_slots
        x_in = R_bore * np.cos(theta)
        y_in = R_bore * np.sin(theta)
        x_out = R_bot * np.cos(theta)
        y_out = R_bot * np.sin(theta)
        femm.mi_drawline(x_in, y_in, x_out, y_out)

    # 2. Place two labels per slot — one on each side of the radial divider.
    for k in range(g.num_slots):
        theta_center = 2 * np.pi * k / g.num_slots
        theta_left = theta_center - quarter_slot_angle
        theta_right = theta_center + quarter_slot_angle

        x_left = R_lbl * np.cos(theta_left)
        y_left = R_lbl * np.sin(theta_left)
        x_right = R_lbl * np.cos(theta_right)
        y_right = R_lbl * np.sin(theta_right)

        # Right half = tooth k go-side
        phase_right = _FSCW_V2_TOOTH_PHASES[k]
        pol_right = _FSCW_V2_TOOTH_POLARITIES[k]
        turns_right = pol_right * g.turns_per_slot

        # Left half = tooth (k-1) return-side (so polarity inverts)
        phase_left = _FSCW_V2_TOOTH_PHASES[(k - 1) % n_teeth]
        pol_left = -_FSCW_V2_TOOTH_POLARITIES[(k - 1) % n_teeth]
        turns_left = pol_left * g.turns_per_slot

        femm.mi_addblocklabel(x_left, y_left)
        femm.mi_selectlabel(x_left, y_left)
        femm.mi_setblockprop(cfg.mat.winding, 1, 0,
                             f"phase_{phase_left}", 0, 1, turns_left)
        femm.mi_clearselected()

        femm.mi_addblocklabel(x_right, y_right)
        femm.mi_selectlabel(x_right, y_right)
        femm.mi_setblockprop(cfg.mat.winding, 1, 0,
                             f"phase_{phase_right}", 0, 1, turns_right)
        femm.mi_clearselected()

    log.debug(
        "Double-layer FSCW winding: %d teeth, pattern=%s, polarities=%s",
        n_teeth, _FSCW_V2_TOOTH_PHASES, _FSCW_V2_TOOTH_POLARITIES,
    )


def build_fscw_v2(cfg: Config, fem_path: str) -> None:
    """Build the FSCW PM model with proper double-layer winding and save it.

    Identical to v1 (`build_fscw`) for stator geometry, rotor geometry, and
    magnet placement. The only difference is the winding stage, which calls
    `_define_circuits_and_coils_double_layer` instead of the single-layer v1
    function — each slot is split radially and carries two coil sides from
    the two adjacent teeth.
    """
    femm = _open_femm()
    femm.openfemm(1 if cfg.solver.hide_window else 0)
    femm.newdocument(cfg.solver.problem_type)
    femm.mi_probdef(
        0,
        cfg.solver.units,
        cfg.solver.symmetry,
        cfg.solver.precision,
        cfg.geom.stack_length_mm,
        cfg.solver.min_angle_deg,
    )
    log.info("FEMM initialised (FSCW v2): planar magnetostatic, depth=%.1f mm",
             cfg.geom.stack_length_mm)

    _define_materials(femm, cfg)
    log.info("Materials defined")

    # Stator (same as IPM / FSCW v1)
    _draw_circle_arcs(femm, cfg.geom.R_stator_outer)
    _draw_circle_arcs(femm, cfg.geom.R_stator_bore)
    _draw_slots(femm, cfg)
    log.info("Stator drawn: OD=%.1f, bore=%.1f, %d slots",
             cfg.geom.R_stator_outer, cfg.geom.R_stator_bore, cfg.geom.num_slots)

    # Rotor: same SPM sector-magnet topology as v1.
    _draw_circle_arcs(femm, cfg.geom.R_rotor_inner)
    _draw_rotor_surface(femm, cfg)
    log.info("Rotor drawn: OD=%.1f, ID=%.1f, %d sector magnets "
             "(t=%.1f mm, arc length=%.1f mm), air-gap=%.2f mm",
             cfg.geom.R_rotor_outer, cfg.geom.R_rotor_inner, cfg.geom.num_poles,
             cfg.geom.magnet_thickness, cfg.geom.magnet_width, cfg.geom.air_gap_mm)

    labels = _place_labels(femm, cfg)
    log.info("Placed %d material labels", len(labels))
    _define_circuits_and_coils_double_layer(femm, cfg)
    log.info("3-phase FSCW circuits + double-layer coil labels assigned "
             "(2 labels x %d slots = %d winding labels)",
             cfg.geom.num_slots, 2 * cfg.geom.num_slots)
    _add_airbox_and_bc(femm, cfg)
    log.info("Air box + Dirichlet BC added")

    femm.mi_saveas(fem_path)
    log.info("Saved FSCW v2 geometry to %s", fem_path)
