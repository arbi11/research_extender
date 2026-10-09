"""12-slot / 10-pole IPM motor geometry build for pyFEMM.

Ported from jupyter_book/Chp3/content/ipm_motor_12slot_10pole.ipynb with the
block-label fixes from lit_survey_c3/plan.md applied.

Public entry point:
    build_ipm(cfg, fem_path) -> None
        Initialises FEMM, builds the geometry, defines materials, assigns
        block labels and circuits, places the air-box and Dirichlet BC,
        and saves the model to `fem_path`. Leaves FEMM open with the model
        loaded so the caller can run the sweep without re-building.
"""

from __future__ import annotations
import logging
from typing import List, Tuple

import numpy as np

from .config import Config


log = logging.getLogger(__name__)

_LabelXY = Tuple[float, float]


def _open_femm():
    """Import femm lazily so non-Windows boxes can at least import the module."""
    import femm  # noqa: WPS433  (import-inside-function is intentional)
    return femm


def _draw_circle_arcs(femm, radius: float, segments_per_quarter: int = 5) -> None:
    """Draw a full circle of given radius as four 90-degree arcs."""
    for i in range(4):
        a0 = np.radians(90.0 * i)
        a1 = np.radians(90.0 * (i + 1))
        x1, y1 = radius * np.cos(a0), radius * np.sin(a0)
        x2, y2 = radius * np.cos(a1), radius * np.sin(a1)
        femm.mi_drawarc(x1, y1, x2, y2, 90.0, segments_per_quarter)


def _draw_slots(femm, cfg: Config) -> None:
    g = cfg.geom
    r_bottom = g.R_stator_bore + g.slot_depth
    half = np.radians(g.slot_angle_deg) / 2.0
    for k in range(g.num_slots):
        theta = 2 * np.pi * k / g.num_slots
        theta_L, theta_R = theta - half, theta + half
        x1, y1 = g.R_stator_bore * np.cos(theta_L), g.R_stator_bore * np.sin(theta_L)
        x2, y2 = g.R_stator_bore * np.cos(theta_R), g.R_stator_bore * np.sin(theta_R)
        x3, y3 = r_bottom * np.cos(theta_L), r_bottom * np.sin(theta_L)
        x4, y4 = r_bottom * np.cos(theta_R), r_bottom * np.sin(theta_R)
        # opening arc at bore
        femm.mi_drawarc(x1, y1, x2, y2, g.slot_angle_deg, 2)
        # radial sides
        femm.mi_drawline(x1, y1, x3, y3)
        femm.mi_drawline(x2, y2, x4, y4)
        # bottom arc
        femm.mi_drawarc(x3, y3, x4, y4, g.slot_angle_deg, 2)


def _draw_magnets(femm, cfg: Config) -> None:
    """Draw each magnet plus its two tangential flux barriers (if enabled).

    Local frame (before rotation) per pole:
        - Magnet rectangle: x in [-L/2, +L/2], y in [-W/2, +W/2].
        - Barrier rectangles (air-filled, group 2 = rotor):
            +y barrier: x in [-L/2, +L/2], y in [+W/2, +W/2 + B]
            -y barrier: x in [-L/2, +L/2], y in [-W/2 - B, -W/2]
        The whole stack (magnet + 2 barriers) is then rotated by theta_mag_deg
        around the magnet centre so the radial axis points outward.
    """
    g = cfg.geom
    R_mag_c = (g.R_rotor_outer + g.R_rotor_inner) / 2.0
    half_L = g.magnet_length / 2.0
    half_W = g.magnet_width / 2.0
    B = g.flux_barrier_width
    if R_mag_c + half_L > g.R_rotor_outer:
        raise ValueError(
            f"Magnet extends beyond rotor OD: {R_mag_c + half_L:.2f} > {g.R_rotor_outer:.2f} mm"
        )
    if R_mag_c - half_L < g.R_rotor_inner:
        raise ValueError(
            f"Magnet extends below rotor ID: {R_mag_c - half_L:.2f} < {g.R_rotor_inner:.2f} mm"
        )
    for p in range(g.num_poles):
        theta_deg = p * (360.0 / g.num_poles) + g.rotor_offset_deg
        cx, cy = R_mag_c * np.cos(np.radians(theta_deg)), R_mag_c * np.sin(np.radians(theta_deg))
        # 1) Magnet rectangle.
        mx1, my1 = -half_L + cx, -half_W + cy
        mx2, my2 = half_L + cx, half_W + cy
        femm.mi_drawrectangle(mx1, my1, mx2, my2)
        # 2) Flux-barrier rectangles, if enabled.
        if B > 0:
            # +y barrier
            femm.mi_drawrectangle(-half_L + cx, half_W + cy,
                                  half_L + cx, half_W + B + cy)
            # -y barrier
            femm.mi_drawrectangle(-half_L + cx, -half_W - B + cy,
                                  half_L + cx, -half_W + cy)
        # 3) Rotate the whole local stack (magnet + barriers) around (cx, cy).
        margin = 0.1
        # selection rectangle big enough to cover magnet + both barriers
        sel_y1, sel_y2 = -half_W - B + cy - margin, half_W + B + cy + margin
        sel_x1, sel_x2 = -half_L + cx - margin, half_L + cx + margin
        femm.mi_selectrectangle(sel_x1, sel_y1, sel_x2, sel_y2, 4)
        femm.mi_moverotate(cx, cy, theta_deg)
        femm.mi_clearselected()


def _define_materials(femm, cfg: Config) -> None:
    femm.mi_getmaterial("Air")
    femm.mi_getmaterial(cfg.mat.stator_steel)
    femm.mi_getmaterial(cfg.mat.winding)
    # NdFeB (custom, taken from pyFEMM manual)
    femm.mi_addmaterial(
        cfg.mat.magnet_name,
        cfg.mat.magnet_mu_r, cfg.mat.magnet_mu_r,   # mu_x, mu_y
        cfg.mat.magnet_Hc_Am,                       # H_c
        0, 0, 0, 0,                                  # J, sigma, lam_d, phi_hmax
        1, 0,                                        # lam_fill, lam_type (not laminated)
        0, 0, 1, 0,                                  # phi_hx, phi_hy, nstr, dwire
    )


def _place_labels(femm, cfg: Config) -> List[_LabelXY]:
    """Place material block labels. Returns list of label coordinates for logging."""
    g = cfg.geom
    labels: List[_LabelXY] = []

    # 1. Stator back iron — between slots
    r = (g.R_stator_outer + (g.R_stator_bore + g.slot_depth)) / 2.0
    a = np.radians(360.0 / g.num_slots / 2.0)  # midway between slot 0 and slot 1
    sx, sy = r * np.cos(a), r * np.sin(a)
    femm.mi_addblocklabel(sx, sy)
    femm.mi_selectlabel(sx, sy)
    femm.mi_setblockprop(cfg.mat.stator_steel, 1, 0, "<None>", 0, 1, 0)
    femm.mi_clearselected()
    labels.append((sx, sy))

    # 2. Air-gap with refined mesh (between stator bore and rotor OD)
    rg = (g.R_stator_bore + g.R_rotor_outer) / 2.0
    femm.mi_addblocklabel(rg, 0.0)
    femm.mi_selectlabel(rg, 0.0)
    femm.mi_setblockprop("Air", 1, cfg.solver.air_gap_mesh_mm, "<None>", 0, 0, 0)
    femm.mi_clearselected()
    labels.append((rg, 0.0))

    # 3. Rotor back iron — midway between two adjacent magnet poles.
    #    Account for rotor_offset_deg so the label lands in steel, not inside
    #    a magnet pocket.
    rb = (g.R_rotor_outer + g.R_rotor_inner) / 2.0
    ab_deg = (360.0 / g.num_poles) / 2.0 + g.rotor_offset_deg
    ab = np.radians(ab_deg)
    rbx, rby = rb * np.cos(ab), rb * np.sin(ab)
    femm.mi_addblocklabel(rbx, rby)
    femm.mi_selectlabel(rbx, rby)
    femm.mi_setblockprop(cfg.mat.rotor_steel, 1, 0, "<None>", 0, 2, 0)
    femm.mi_clearselected()
    labels.append((rbx, rby))

    # 4. Rotor inner bore (air at origin)
    femm.mi_addblocklabel(0.0, 0.0)
    femm.mi_selectlabel(0.0, 0.0)
    femm.mi_setblockprop("Air", 1, 0, "<None>", 0, 0, 0)
    femm.mi_clearselected()
    labels.append((0.0, 0.0))

    # 5. Magnets (centre of each pole) and 6. Flux-barrier air pockets at each
    #    magnet's tangential ends. Both are in group 2 so they move with the rotor.
    R_mag_c = (g.R_rotor_outer + g.R_rotor_inner) / 2.0
    half_W = g.magnet_width / 2.0
    B = g.flux_barrier_width
    for p in range(g.num_poles):
        theta_deg = p * (360.0 / g.num_poles) + g.rotor_offset_deg
        theta = np.radians(theta_deg)
        cx, cy = R_mag_c * np.cos(theta), R_mag_c * np.sin(theta)

        # Magnet label at pole centre with radial magnetisation (alternating)
        mag_dir = theta_deg if p % 2 == 0 else theta_deg + 180.0
        femm.mi_addblocklabel(cx, cy)
        femm.mi_selectlabel(cx, cy)
        femm.mi_setblockprop(cfg.mat.magnet_name, 1, 0, "<None>", mag_dir, 2, 0)
        femm.mi_clearselected()
        labels.append((cx, cy))

        # Flux-barrier labels: tangentially offset from pole centre by
        # (half_W + B/2) in the LOCAL frame, rotated by theta_deg into global.
        if B > 0:
            offset = half_W + B / 2.0
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            # +y local -> global direction is (-sin_t, +cos_t)
            bxp = cx - offset * sin_t
            byp = cy + offset * cos_t
            bxm = cx + offset * sin_t
            bym = cy - offset * cos_t
            for bx, by in [(bxp, byp), (bxm, bym)]:
                femm.mi_addblocklabel(bx, by)
                femm.mi_selectlabel(bx, by)
                femm.mi_setblockprop("Air", 1, 0, "<None>", 0, 2, 0)
                femm.mi_clearselected()
                labels.append((bx, by))

    return labels


def _define_circuits_and_coils(femm, cfg: Config) -> None:
    # Each phase as series circuit, initial current 0 (overwritten per sweep step).
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


def _add_airbox_and_bc(femm, cfg: Config) -> None:
    size = cfg.geom.R_stator_outer * cfg.solver.boundary_size_factor
    femm.mi_drawrectangle(-size, -size, size, size)
    femm.mi_addboundprop("ZeroA", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    for x, y in [(-size, 0.0), (size, 0.0), (0.0, -size), (0.0, size)]:
        femm.mi_selectsegment(x, y)
    femm.mi_setsegmentprop("ZeroA", 0, 1, 0, 0)
    femm.mi_clearselected()
    # Background air label in a corner, coarse mesh
    bx, by = size * 0.75, size * 0.75
    femm.mi_addblocklabel(bx, by)
    femm.mi_selectlabel(bx, by)
    femm.mi_setblockprop("Air", 1, 1.0, "<None>", 0, 0, 0)
    femm.mi_clearselected()


def build_ipm(cfg: Config, fem_path: str) -> None:
    """Build the IPM model and save to `fem_path`. Leaves FEMM open."""
    femm = _open_femm()
    # bHide=1 keeps the FEMM main window hidden (per pyFEMM manual). Useful
    # for headless / parallel sweeps.
    femm.openfemm(1 if cfg.solver.hide_window else 0)
    femm.newdocument(cfg.solver.problem_type)
    femm.mi_probdef(
        0,                      # 0 = magnetostatic frequency
        cfg.solver.units,
        cfg.solver.symmetry,
        cfg.solver.precision,
        cfg.geom.stack_length_mm,
        cfg.solver.min_angle_deg,
    )

    log.info("FEMM initialised: planar magnetostatic, units=%s, depth=%.1f mm",
             cfg.solver.units, cfg.geom.stack_length_mm)

    _define_materials(femm, cfg)
    log.info("Materials defined")

    # Stator
    _draw_circle_arcs(femm, cfg.geom.R_stator_outer)
    _draw_circle_arcs(femm, cfg.geom.R_stator_bore)
    _draw_slots(femm, cfg)
    log.info("Stator drawn: OD=%.1f, bore=%.1f, %d slots",
             cfg.geom.R_stator_outer, cfg.geom.R_stator_bore, cfg.geom.num_slots)

    # Rotor
    _draw_circle_arcs(femm, cfg.geom.R_rotor_outer)
    _draw_circle_arcs(femm, cfg.geom.R_rotor_inner)
    _draw_magnets(femm, cfg)
    log.info("Rotor drawn: OD=%.1f, ID=%.1f, %d poles, air-gap=%.2f mm",
             cfg.geom.R_rotor_outer, cfg.geom.R_rotor_inner, cfg.geom.num_poles,
             cfg.geom.air_gap_mm)

    labels = _place_labels(femm, cfg)
    log.info("Placed %d material labels", len(labels))
    _define_circuits_and_coils(femm, cfg)
    log.info("3-phase circuits + %d coil labels assigned", cfg.geom.num_slots)
    _add_airbox_and_bc(femm, cfg)
    log.info("Air box + Dirichlet BC added")

    femm.mi_saveas(fem_path)
    log.info("Saved geometry to %s", fem_path)
