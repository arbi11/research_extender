"""12-slot / 10-pole FSCW PM motor geometry build for pyFEMM.

Surface-mount magnets (rectangular approximation of arcs), concentrated tooth
windings. The stator drawing reuses IPM helpers (`_draw_circle_arcs`,
`_draw_slots`); the rotor and magnet placement are FSCW-specific.

Public entry point:
    build_fscw(cfg, fem_path) -> None
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


def _define_circuits_and_coils(femm, cfg: Config) -> None:
    """FSCW double-layer winding: same coil-per-slot layout as IPM but with
    the FSCW phase/polarity pattern from `FSCWWindingPattern`.
    """
    femm.mi_addcircprop("phase_A", 0.0, 1)
    femm.mi_addcircprop("phase_B", 0.0, 1)
    femm.mi_addcircprop("phase_C", 0.0, 1)

    g = cfg.geom
    w = cfg.winding
    slot_label_r = (g.R_stator_bore + (g.R_stator_bore + g.slot_depth)) / 2.0
    for k in range(g.num_slots):
        theta = 2 * np.pi * k / g.num_slots
        x, y = slot_label_r * np.cos(theta), slot_label_r * np.sin(theta)
        phase = w.phases[k]
        polarity = w.polarities[k]
        turns = g.turns_per_slot * polarity
        femm.mi_addblocklabel(x, y)
        femm.mi_selectlabel(x, y)
        femm.mi_setblockprop(cfg.mat.winding, 1, 0, f"phase_{phase}", 0, 1, turns)
        femm.mi_clearselected()


def build_fscw(cfg: Config, fem_path: str) -> None:
    """Build the FSCW PM model and save to `fem_path`. Leaves FEMM open."""
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
    log.info("FEMM initialised (FSCW): planar magnetostatic, depth=%.1f mm",
             cfg.geom.stack_length_mm)

    _define_materials(femm, cfg)
    log.info("Materials defined")

    # Stator (same as IPM)
    _draw_circle_arcs(femm, cfg.geom.R_stator_outer)
    _draw_circle_arcs(femm, cfg.geom.R_stator_bore)
    _draw_slots(femm, cfg)
    log.info("Stator drawn: OD=%.1f, bore=%.1f, %d slots",
             cfg.geom.R_stator_outer, cfg.geom.R_stator_bore, cfg.geom.num_slots)

    # Rotor: rotor inner bore as a full circle; rotor outer boundary is built
    # piecewise by _draw_rotor_surface as alternating magnet outer arcs and
    # inter-magnet steel arcs (proper SPM topology — magnet face = rotor surface).
    _draw_circle_arcs(femm, cfg.geom.R_rotor_inner)
    _draw_rotor_surface(femm, cfg)
    log.info("Rotor drawn: OD=%.1f, ID=%.1f, %d sector magnets "
             "(t=%.1f mm, arc length=%.1f mm), air-gap=%.2f mm",
             cfg.geom.R_rotor_outer, cfg.geom.R_rotor_inner, cfg.geom.num_poles,
             cfg.geom.magnet_thickness, cfg.geom.magnet_width, cfg.geom.air_gap_mm)

    labels = _place_labels(femm, cfg)
    log.info("Placed %d material labels", len(labels))
    _define_circuits_and_coils(femm, cfg)
    log.info("3-phase FSCW circuits + %d coil labels assigned", cfg.geom.num_slots)
    _add_airbox_and_bc(femm, cfg)
    log.info("Air box + Dirichlet BC added")

    femm.mi_saveas(fem_path)
    log.info("Saved FSCW geometry to %s", fem_path)
