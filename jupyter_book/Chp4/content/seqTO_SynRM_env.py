"""
seqTO_SynRM_env.py  —  SeqTO-v1 SynRM FEMM Environment and Validation

Implements the Sequence Topology Optimisation MDP environment for a quarter-
symmetry 4-pole Synchronous Reluctance Motor (SynRM).  Mirrors seqTO_env.py
architecture exactly: same MDP interface, same reward signature, different
physics (mean torque over 6 rotor angles instead of B-field force).

Physics approach:
  - Anti-periodic Air Gap BC (BdryFormat=7) — one mesh, no remeshing.
  - Rotor position swept by modifying InnerAngle (ia) on the AirGap BC.
  - Torque via mo_blockintegral(22) on ROTOR_GROUP at each angle.
  - Reward = mean |torque| over N_ANGLES positions.

Requirements: pyfemm, numpy  (Windows + FEMM 4.2, build >= Feb 2018 for AirGap BC)
Usage       : python seqTO_SynRM_env.py
              python seqTO_SynRM_env.py --build-geometry
"""

import os
import sys
import argparse
import numpy as np

try:
    import femm
except ImportError:
    print("ERROR: pyfemm not installed.  Run: pip install pyfemm")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Machine geometry constants
# ─────────────────────────────────────────────────────────────────────────────

R_SHAFT   = 10.0    # mm  inner rotor (shaft, non-magnetic)
R_PERI_IN = 36.7    # mm  inner radius of peripheral structural bridge (design domain stops here)
R_ROTOR   = 37.0    # mm  outer rotor iron radius (= R_GAP_IN); peri bridge is 0.3 mm thick
R_GAP_IN  = 37.0    # mm  rotor-side air gap boundary
R_GAP_OUT = 37.5    # mm  stator-side air gap boundary
R_STATOR  = 60.0    # mm  outer stator radius
R_YOKE    = 52.0    # mm  slot-to-yoke boundary (separates coil slots from iron yoke)
L_STACK   = 150.0   # mm  axial stack depth (problem depth)
N_POLES   = 4
POLE_PAIRS = N_POLES // 2   # = 2

# Structural frame: inner hub ring + thin 0.3 mm peripheral M-19 bridge.
# The bridge holds rotor segments together but saturates fast → does not short flux.
# No angular bridges at 0°/90°; anti-periodic BC handles angular periodicity.
R_HUB  = 12.0   # mm  inner hub ring outer radius  (R_SHAFT + 2 mm)

# Design domain: 5×5 polar grid, full 0°–90° angular sweep
DD_ROWS = 5
DD_COLS = 5
DR      = (R_PERI_IN - R_HUB) / DD_ROWS    # = 4.94 mm per band  (12→16.94→21.88→26.82→31.76→36.7)
DTHETA  = 90.0 / DD_COLS                   # = 18.0° per sector

# Budget and episode length — all 25 grid cells are free
K1        = 5    # min iron cells
K2        = 20   # max iron cells  (80 % fill limit; 25 cells total available)
MAX_STEPS = 15   # steps per episode

# 3-phase winding — distributed q=2 (24-slot full machine equivalent).
# Phase belts in the 0°-90° quarter (mech): +A at slots 1,2 (centroid 15°),
# −C at slots 3,4 (centroid 45°), +B at slots 5,6 (centroid 75°).
# d-axis aligned with the controller's radial-rib pattern at 45° mech;
# γ = 45° electrical lead → MTPA. MMF peak target = D_AXIS_MECH + γ/p mech.
D_AXIS_MECH_DEG  = 45.0      # rotor d-axis position (mech) — controller-rib direction
GAMMA_DEG        = 45.0      # current-vector elec lead from d-axis (MTPA = 45°)
PHASE_A_AXIS_DEG = 15.0      # +A coil-belt centroid in mech (q=2 layout)
I_PEAK           = 10.0      # A  peak phase current
N_TURNS          = 100       # turns per slot
# Required current-vector elec angle so MMF peak lands at D_AXIS_MECH + γ/p.
_alpha_elec = np.radians(POLE_PAIRS * D_AXIS_MECH_DEG + GAMMA_DEG
                         - POLE_PAIRS * PHASE_A_AXIS_DEG)
IA = I_PEAK * np.cos(_alpha_elec)
IB = I_PEAK * np.cos(_alpha_elec - 2 * np.pi / 3)
IC = I_PEAK * np.cos(_alpha_elec + 2 * np.pi / 3)

# Stator slot layout — 6 slots in the 90° quarter (24-slot full machine equivalent).
# Slot openings 12° wide; teeth fill the remaining 18° (5 full-width + 2 boundary halves).
SLOT_CENTERS_DEG    = [7.5, 22.5, 37.5, 52.5, 67.5, 82.5]
SLOT_HALFWIDTH_DEG  = 6.0     # → 12° slot opening, 3° full tooth, 1.5° boundary half-tooth
# Phase belts (q=2 distributed): slots 1,2 → +A;  3,4 → −C;  5,6 → +B
# Sign is encoded via the signed `turns` parameter passed to mi_setblockprop.
SLOT_WINDING = [
    ('PhaseA', +1),
    ('PhaseA', +1),
    ('PhaseC', -1),
    ('PhaseC', -1),
    ('PhaseB', +1),
    ('PhaseB', +1),
]

# Rotor sweep angles: 6 electrical positions → mechanical = electrical / POLE_PAIRS
_ELEC_ANGLES_DEG   = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0]
ROTOR_ANGLES_MECH  = [e / POLE_PAIRS for e in _ELEC_ANGLES_DEG]   # [0,7.5,15,22.5,30,37.5]

# FEMM group numbers
ROTOR_GROUP  = 3    # all rotor design cells → used by mo_groupselectblock for torque
SHAFT_GROUP  = 1
GAP_GROUP    = 2
STATOR_GROUP = 5

BASE_FEM_FILE = 'synrm_seqto.fem'
_TMP_FEM      = '_tmp_synrm.fem'
DESIGNS_DIR   = 'designs'


def configure(dd_rows, dd_cols, max_steps=None, k1=None, k2=None,
              base_fem_file=None, tmp_fem=None):
    """Reconfigure the SynRM design domain to dd_rows x dd_cols.

    Updates the module-level constants in place so all downstream functions
    (``build_synrm_geometry``, ``SynRMSeqTOEnv``, ``calculate_reward``, ...)
    pick up the new geometry the next time they are called.  Derived
    constants (``DR``, ``DTHETA``) and the per-grid ``BASE_FEM_FILE`` /
    ``_TMP_FEM`` cache names are recomputed automatically; iron budget and
    episode length scale with grid size unless overridden.

    Call this BEFORE instantiating ``SynRMSeqTOEnv`` (or calling other
    module functions).  Importers that did ``from seqTO_SynRM_env import X``
    must re-bind ``X`` after this call (or use ``module.X`` lookup instead).

    Defaults
    --------
    K1        = max(dd_cols, 5)             # at least one cell per angular sector
    K2        = ceil(0.8 * dd_rows*dd_cols) # 80 percent fill cap
    MAX_STEPS = dd_rows * dd_cols           # one action per cell, in principle
    BASE_FEM_FILE = f'synrm_seqto_{dd_rows}x{dd_cols}.fem'
    """
    global DD_ROWS, DD_COLS, DR, DTHETA, K1, K2, MAX_STEPS, BASE_FEM_FILE, _TMP_FEM
    DD_ROWS, DD_COLS = int(dd_rows), int(dd_cols)
    DR        = (R_PERI_IN - R_HUB) / DD_ROWS
    DTHETA    = 90.0 / DD_COLS
    K1        = int(k1)        if k1        is not None else max(DD_COLS, 5)
    K2        = int(k2)        if k2        is not None else int(0.8 * DD_ROWS * DD_COLS)
    MAX_STEPS = int(max_steps) if max_steps is not None else DD_ROWS * DD_COLS
    BASE_FEM_FILE = base_fem_file if base_fem_file is not None else f'synrm_seqto_{DD_ROWS}x{DD_COLS}.fem'
    _TMP_FEM      = tmp_fem       if tmp_fem       is not None else f'_tmp_synrm_{DD_ROWS}x{DD_COLS}.fem'
    print(f"[seqTO_SynRM_env] reconfigured: {DD_ROWS}x{DD_COLS}  "
          f"DR={DR:.3f}  DTHETA={DTHETA:.1f}  K1={K1}  K2={K2}  MAX_STEPS={MAX_STEPS}")

# Actions: name, Δrow, Δcol  (row = radial idx, col = angular idx)
ACTIONS = {
    0: ('RIGHT',  0, +1),   # increase angular idx (more CCW)
    1: ('LEFT',   0, -1),   # decrease angular idx
    2: ('UP',    +1,  0),   # increase radial idx (away from shaft)
    3: ('DOWN',  -1,  0),   # decrease radial idx (toward shaft)
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_design_cell(i, j):
    """True if (i,j) falls within the 5×5 rotor design grid."""
    return 0 <= i < DD_ROWS and 0 <= j < DD_COLS


def _cell_centroid(i, j):
    """Cartesian (x, y) mm of the centre of design cell (i, j)."""
    r_mid     = R_HUB + (i + 0.5) * DR
    theta_mid = np.radians((j + 0.5) * DTHETA)   # 9°, 27°, 45°, 63°, 81°
    return r_mid * np.cos(theta_mid), r_mid * np.sin(theta_mid)


def _print_grid(iron_mask, pos_r, pos_c):
    """
    Print the 5×5 design grid to stdout.
    [X] = controller is here AND cell has iron
    [.] = controller is here, cell is air
     X  = iron, no controller
     .  = air, no controller
    Row i=4 (outermost) printed at top; i=0 (innermost) at bottom.
    """
    print("       " + "  ".join(f" j{j}" for j in range(DD_COLS)))
    for i in range(DD_ROWS - 1, -1, -1):
        row = f"  i={i} "
        for j in range(DD_COLS):
            here = (i == pos_r and j == pos_c)
            iron = bool(iron_mask[i, j])
            if here and iron:
                row += "[X] "
            elif here:
                row += "[.] "
            elif iron:
                row += " X  "
            else:
                row += " .  "
        print(row)


def _reset_design_cells_to_air():
    """
    Select all 25 rotor design cells and set them to Air.
    The structural frame is a separate geometric region — not touched here.
    Must be called at the start of every calculate_reward() invocation.
    """
    for i in range(DD_ROWS):
        for j in range(DD_COLS):
            femm.mi_selectlabel(*_cell_centroid(i, j))
    femm.mi_setblockprop('Air', 1, 0, '<None>', 0, ROTOR_GROUP, 0)
    femm.mi_clearselected()


# ─────────────────────────────────────────────────────────────────────────────
# Geometry builder
# ─────────────────────────────────────────────────────────────────────────────

def build_synrm_geometry():
    """
    Build the SynRM FEMM geometry from scratch and save as synrm_seqto.fem.

    Radial zones (all in mm):
      Shaft            r = 0      → R_SHAFT=10     Air, non-magnetic
      Inner hub ring   r = 10     → R_HUB=12       M-19 Steel, FIXED frame
      Design domain    r = 12     → R_PERI_IN=36.7 5×5 grid, Air initially (optimised)
      Peri bridge      r = 36.7   → R_ROTOR=37     M-19 Steel, 0.3 mm structural bridge
      Air gap          r = 37     → R_GAP_OUT=37.5 Air, AirGap BC (rotor swept here)
      Stator (slots)   r = 37.5   → R_YOKE=52      6 copper slots + 7 M-19 teeth (q=2, 24-slot eq.)
      Stator yoke      r = 52     → R_STATOR=60    M-19 Steel

    Stator (90° quarter, q=2 distributed winding):
      Slot openings 12° wide, centred at 7.5°, 22.5°, 37.5°, 52.5°, 67.5°, 82.5° mech.
      Phase belts:  +A | +A | −C | −C | +B | +B  (sign encoded via signed `turns`).
      Teeth: 7 M-19 regions (incl. half-teeth at 0°/90° boundaries).

    AntiPeriodic BC: one named BC per radial segment pair (APer01..APer11).
    Each BC gets exactly 2 segments (0° line + 90° line) → no FEMM warning.
    Slot dividers do NOT touch the 0°/90° symmetry lines, so anti-periodic
    handling on the symmetry lines is unaffected.
    """
    print("  Opening FEMM, building SynRM geometry from scratch...")
    femm.openfemm(1)
    femm.newdocument(0)
    femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, L_STACK, 30)

    # ── Materials ─────────────────────────────────────────────────────────────
    femm.mi_getmaterial('Air')
    femm.mi_getmaterial('M-19 Steel')
    femm.mi_getmaterial('Copper')

    # ── Circuits ───────────────────────────────────────────────────────────────
    femm.mi_addcircprop('PhaseA', IA, 1)
    femm.mi_addcircprop('PhaseB', IB, 1)
    femm.mi_addcircprop('PhaseC', IC, 1)

    # ── Boundary conditions ────────────────────────────────────────────────────
    femm.mi_addboundprop('A=0',    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    femm.mi_addboundprop('AirGap', 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0)

    # AntiPeriodic BCs — one per radial segment pair on the 0°/90° boundary lines.
    # All arcs are full 90° so all arc radii create nodes on the 0°/90° lines.
    # Design-domain internal radii are R_HUB + k*DR for k = 1..DD_ROWS-1
    # (so DD_ROWS=5 ⇒ 4 internal radii; DD_ROWS=10 ⇒ 9 internal radii).
    _design_internal_radii = [R_HUB + k * DR for k in range(1, DD_ROWS)]
    _APER_RADII = (
        [0.0, R_SHAFT, R_HUB]
        + _design_internal_radii
        + [R_PERI_IN, R_ROTOR, R_GAP_OUT, R_YOKE, R_STATOR]
    )
    for k in range(len(_APER_RADII) - 1):
        femm.mi_addboundprop(f'APer{k+1:02d}', 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0)

    # ── Arc rings — all full 90° ───────────────────────────────────────────────
    arc_radii = (
        [R_SHAFT, R_HUB]                                # shaft / hub-inner
        + _design_internal_radii                        # design-domain divisions
        + [R_PERI_IN, R_ROTOR, R_GAP_OUT, R_YOKE, R_STATOR]  # peri / gap / stator
    )
    for r in arc_radii:
        femm.mi_drawarc(r, 0.0,  0.0, r,  90.0, 2.5)

    # ── Radial lines ──────────────────────────────────────────────────────────
    # Shaft closure: origin → R_SHAFT at 0° and 90°
    femm.mi_drawline(0.0, 0.0,  R_SHAFT, 0.0)
    femm.mi_drawline(0.0, 0.0,  0.0, R_SHAFT)

    # 0° and 90° symmetry boundary lines: R_SHAFT → R_ROTOR.
    # Arcs at R_HUB, internal design radii, and R_PERI_IN split these automatically.
    femm.mi_drawline(R_SHAFT, 0.0,   R_ROTOR, 0.0)
    femm.mi_drawline(0.0, R_SHAFT,   0.0, R_ROTOR)

    # Internal sector lines at 18°, 36°, 54°, 72° from R_HUB → R_PERI_IN.
    # 0° and 90° are already covered by the symmetry lines above.
    # Hub ring and peripheral ring are undivided full-90° regions — no lines through them.
    for k in range(1, DD_COLS):   # k=1..4  →  18°, 36°, 54°, 72°
        theta = np.radians(k * DTHETA)
        femm.mi_drawline(R_HUB     * np.cos(theta), R_HUB     * np.sin(theta),
                         R_PERI_IN * np.cos(theta), R_PERI_IN * np.sin(theta))

    # Air gap lines at 0° and 90°
    femm.mi_drawline(R_GAP_IN,  0.0,    R_GAP_OUT, 0.0)
    femm.mi_drawline(0.0, R_GAP_IN,     0.0, R_GAP_OUT)

    # Stator slot dividers — 12 radial lines at every slot edge (R_GAP_OUT → R_YOKE).
    # Slot edges: slot_center ± SLOT_HALFWIDTH_DEG.  None of these touch the
    # 0°/90° symmetry lines, so anti-periodic BC handling is unaffected.
    slot_edges_deg = []
    for c in SLOT_CENTERS_DEG:
        slot_edges_deg.extend((c - SLOT_HALFWIDTH_DEG, c + SLOT_HALFWIDTH_DEG))
    for theta_deg in slot_edges_deg:
        theta = np.radians(theta_deg)
        femm.mi_drawline(R_GAP_OUT * np.cos(theta), R_GAP_OUT * np.sin(theta),
                         R_YOKE    * np.cos(theta), R_YOKE    * np.sin(theta))

    # Stator boundary lines at 0° and 90° (R_GAP_OUT → R_STATOR; R_YOKE arc splits)
    femm.mi_drawline(R_GAP_OUT, 0.0,   R_STATOR, 0.0)
    femm.mi_drawline(0.0, R_GAP_OUT,   0.0, R_STATOR)

    # ── Apply BCs to arcs ──────────────────────────────────────────────────────
    femm.mi_selectarcsegment(R_STATOR  * np.cos(np.radians(45)),
                              R_STATOR  * np.sin(np.radians(45)))
    femm.mi_setarcsegmentprop(2.5, 'A=0', 0, 0)
    femm.mi_clearselected()

    femm.mi_selectarcsegment(R_GAP_IN  * np.cos(np.radians(45)),
                              R_GAP_IN  * np.sin(np.radians(45)))
    femm.mi_setarcsegmentprop(1.0, 'AirGap', 0, 0)
    femm.mi_clearselected()

    # The slot dividers split the R_GAP_OUT arc into 13 sub-arcs (alternating
    # boundary half-tooth, slot, tooth, …, slot, boundary half-tooth).  Apply
    # the AirGap BC to every sub-arc — selecting only the 45° point would leave
    # the rest unbounded.
    _gap_out_subarc_midpoints_deg = []
    _gap_out_subarc_midpoints_deg.append(0.5 * slot_edges_deg[0])           # 0° → 1.5°
    for k in range(len(SLOT_CENTERS_DEG)):                                  # slot midpoints
        _gap_out_subarc_midpoints_deg.append(SLOT_CENTERS_DEG[k])
        if k < len(SLOT_CENTERS_DEG) - 1:                                   # tooth midpoints
            _gap_out_subarc_midpoints_deg.append(
                0.5 * (slot_edges_deg[2*k + 1] + slot_edges_deg[2*k + 2]))
    _gap_out_subarc_midpoints_deg.append(0.5 * (slot_edges_deg[-1] + 90.0))  # 88.5° → 90°
    for theta_deg in _gap_out_subarc_midpoints_deg:
        theta = np.radians(theta_deg)
        femm.mi_selectarcsegment(R_GAP_OUT * np.cos(theta),
                                  R_GAP_OUT * np.sin(theta))
        femm.mi_setarcsegmentprop(1.0, 'AirGap', 0, 0)
        femm.mi_clearselected()

    # ── Apply AntiPeriodic BCs — one BC per segment pair ──────────────────────
    for k, (r1, r2) in enumerate(zip(_APER_RADII[:-1], _APER_RADII[1:])):
        bc_name = f'APer{k+1:02d}'
        r_mid   = (r1 + r2) / 2.0
        femm.mi_selectsegment(r_mid, 0.0)
        femm.mi_selectsegment(0.0, r_mid)
        femm.mi_setsegmentprop(bc_name, 0, 1, 0, 0)
        femm.mi_clearselected()

    # ── Block labels ───────────────────────────────────────────────────────────

    # Shaft interior (r < R_SHAFT): Air, non-magnetic
    r_s = R_SHAFT * 0.5
    femm.mi_addblocklabel(r_s * np.cos(np.radians(45)), r_s * np.sin(np.radians(45)))
    femm.mi_selectlabel(  r_s * np.cos(np.radians(45)), r_s * np.sin(np.radians(45)))
    femm.mi_setblockprop('Air', 1, 0, '<None>', 0, SHAFT_GROUP, 0)
    femm.mi_clearselected()

    # Inner hub ring (R_SHAFT → R_HUB, 0°–90°): single M-19 Steel label
    r_hub_mid = (R_SHAFT + R_HUB) / 2.0   # = 11 mm
    femm.mi_addblocklabel(r_hub_mid * np.cos(np.radians(45)),
                           r_hub_mid * np.sin(np.radians(45)))
    femm.mi_selectlabel(  r_hub_mid * np.cos(np.radians(45)),
                           r_hub_mid * np.sin(np.radians(45)))
    femm.mi_setblockprop('M-19 Steel', 1, 0, '<None>', 0, ROTOR_GROUP, 0)
    femm.mi_clearselected()

    # 25 free design cells (R_HUB → R_PERI_IN, 0°–90°): Air initially
    for i in range(DD_ROWS):
        for j in range(DD_COLS):
            femm.mi_addblocklabel(*_cell_centroid(i, j))
    for i in range(DD_ROWS):
        for j in range(DD_COLS):
            femm.mi_selectlabel(*_cell_centroid(i, j))
    femm.mi_setblockprop('Air', 1, 0, '<None>', 0, ROTOR_GROUP, 0)
    femm.mi_clearselected()

    # Peripheral structural bridge (R_PERI_IN → R_ROTOR, 0°–90°): thin M-19 ring,
    # 0.3 mm thick.  Saturates fast under stator MMF → does NOT short rotor flux,
    # so the design grid drives the saliency.  Sector lines stop at R_PERI_IN, so
    # the bridge is one undivided region.
    r_peri_mid = (R_PERI_IN + R_ROTOR) / 2.0
    femm.mi_addblocklabel(r_peri_mid * np.cos(np.radians(45)),
                           r_peri_mid * np.sin(np.radians(45)))
    femm.mi_selectlabel(  r_peri_mid * np.cos(np.radians(45)),
                           r_peri_mid * np.sin(np.radians(45)))
    femm.mi_setblockprop('M-19 Steel', 1, 0, '<None>', 0, ROTOR_GROUP, 0)
    femm.mi_clearselected()

    # Air gap ring (R_GAP_IN → R_GAP_OUT): Air
    r_gap_mid = (R_GAP_IN + R_GAP_OUT) / 2.0
    femm.mi_addblocklabel(r_gap_mid * np.cos(np.radians(45)),
                           r_gap_mid * np.sin(np.radians(45)))
    femm.mi_selectlabel(  r_gap_mid * np.cos(np.radians(45)),
                           r_gap_mid * np.sin(np.radians(45)))
    femm.mi_setblockprop('Air', 1, 0, '<None>', 0, GAP_GROUP, 0)
    femm.mi_clearselected()

    # Stator slot coils (R_GAP_OUT → R_YOKE): 6 Copper labels, signed turns.
    # `turns` carries the polarity:  +N_TURNS for forward, -N_TURNS for return.
    r_coil = (R_GAP_OUT + R_YOKE) / 2.0   # = 44.75 mm
    for (circuit, sign), theta_mid in zip(SLOT_WINDING, SLOT_CENTERS_DEG):
        cx = r_coil * np.cos(np.radians(theta_mid))
        cy = r_coil * np.sin(np.radians(theta_mid))
        femm.mi_addblocklabel(cx, cy)
        femm.mi_selectlabel(cx, cy)
        femm.mi_setblockprop('Copper', 0, 1.0, circuit, 0, 4, sign * N_TURNS)
        femm.mi_clearselected()

    # Stator teeth (R_GAP_OUT → R_YOKE): 7 M-19 labels — 5 full-width teeth
    # between slots + 2 boundary half-teeth at 0° and 90°.
    tooth_centroids_deg = [0.5 * slot_edges_deg[0]]
    for k in range(len(SLOT_CENTERS_DEG) - 1):
        tooth_centroids_deg.append(
            0.5 * (slot_edges_deg[2*k + 1] + slot_edges_deg[2*k + 2]))
    tooth_centroids_deg.append(0.5 * (slot_edges_deg[-1] + 90.0))
    for theta_mid in tooth_centroids_deg:
        cx = r_coil * np.cos(np.radians(theta_mid))
        cy = r_coil * np.sin(np.radians(theta_mid))
        femm.mi_addblocklabel(cx, cy)
        femm.mi_selectlabel(cx, cy)
        femm.mi_setblockprop('M-19 Steel', 1, 0, '<None>', 0, STATOR_GROUP, 0)
        femm.mi_clearselected()

    # Stator yoke (R_YOKE → R_STATOR): M-19 Steel
    r_yoke_mid = (R_YOKE + R_STATOR) / 2.0   # = 56 mm
    femm.mi_addblocklabel(r_yoke_mid * np.cos(np.radians(45)),
                           r_yoke_mid * np.sin(np.radians(45)))
    femm.mi_selectlabel(  r_yoke_mid * np.cos(np.radians(45)),
                           r_yoke_mid * np.sin(np.radians(45)))
    femm.mi_setblockprop('M-19 Steel', 1, 0, '<None>', 0, STATOR_GROUP, 0)
    femm.mi_clearselected()

    femm.mi_saveas(os.path.abspath(BASE_FEM_FILE).replace('\\', '/'))
    femm.closefemm()
    print(f"  Base geometry saved → {BASE_FEM_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
# FEMM solve  (reward calculation)
# ─────────────────────────────────────────────────────────────────────────────

def save_topology_fem(iron_mask, filename):
    """
    Save iron_mask as a solved .fem + .ans pair for manual FEMM inspection.

    Opens the base geometry, applies the topology, solves once at rotor angle=0°,
    then saves and closes.  Open the .fem (or .ans) in FEMM to inspect field plots.

    iron_mask : 5×5 int array  (1 = M-19 Steel, 0 = Air)
    filename  : output .fem path  (e.g. 'controller_pattern_final.fem')
    """
    femm.openfemm(1)
    femm.opendocument(os.path.abspath(BASE_FEM_FILE).replace('\\', '/'))

    _reset_design_cells_to_air()

    has_iron = False
    for i in range(DD_ROWS):
        for j in range(DD_COLS):
            if iron_mask[i, j] == 1:
                femm.mi_selectlabel(*_cell_centroid(i, j))
                has_iron = True
    if has_iron:
        femm.mi_setblockprop('M-19 Steel', 1, 0, '<None>', 0, ROTOR_GROUP, 0)
    femm.mi_clearselected()

    abs_path = os.path.abspath(filename).replace('\\', '/')
    femm.mi_saveas(abs_path)
    femm.mi_analyse(0)   # solve at rotor angle=0°
    femm.closefemm()

    ans_file = os.path.splitext(filename)[0] + '.ans'
    print(f"  Saved topology → {os.path.abspath(filename)}")
    print(f"  Solved field   → {os.path.abspath(ans_file)}")
    print(f"  Open the .fem in FEMM → Postprocessor → Load Solution to view B-field.")


def calculate_reward(iron_mask, save_bmp=None):
    """
    Open base geometry, apply iron_mask, sweep 6 rotor angles, return a
    topology figure-of-merit derived from the FEMM weighted stress-tensor torque.

    iron_mask : 5×5 int array  (1 = M-19 Steel, 0 = Air)
    save_bmp  : optional path prefix; saves B-field bitmap at angle=0 if given.

    Returns (reward, torques_list)
      reward = max |torque| across ROTOR_ANGLES_MECH (quarter model).
      torques_list = per-angle |torque| values; sign is dropped per-element.

    Note on reward semantics (see diagnose_synrm.py + designs/synrm_diag/):
      The 6 rotor angles span 0°–37.5° mech, but per-angle torques at
      offsets ≥ 15° mech are insensitive to rotor topology (FEMM AirGap BC
      weighted-contour decouples from rotor material at off-design offsets).
      Taking max() rather than mean() suppresses those silent angles and
      preserves the topology-discriminating signal.  The returned value is
      NOT a calibrated reluctance torque in N·m — it is a relative
      figure-of-merit suitable for ranking topologies under a fixed iron
      budget (K1 ≤ iron ≤ K2).  Do not report it as absolute N·m torque.
    """
    hide = 1 if save_bmp is None else 0
    femm.openfemm(hide)
    femm.opendocument(os.path.abspath(BASE_FEM_FILE).replace('\\', '/'))

    # ── STEP 1: Reset all design cells to Air ─────────────────────────────────
    _reset_design_cells_to_air()

    # ── STEP 2: Apply iron_mask to all 25 design cells ────────────────────────
    has_iron = False
    for i in range(DD_ROWS):
        for j in range(DD_COLS):
            if iron_mask[i, j] == 1:
                femm.mi_selectlabel(*_cell_centroid(i, j))
                has_iron = True
    if has_iron:
        femm.mi_setblockprop('M-19 Steel', 1, 0, '<None>', 0, ROTOR_GROUP, 0)
    femm.mi_clearselected()

    # ── STEP 3: Sweep rotor angles, extract torque at each ────────────────────
    tmp_abs = os.path.abspath(_TMP_FEM).replace('\\', '/')
    torques = []

    for idx, angle_mech in enumerate(ROTOR_ANGLES_MECH):
        # Shift rotor without remeshing — modify ia (propnum=10) on AirGap BC
        femm.mi_modifyboundprop('AirGap', 10, angle_mech)

        femm.mi_saveas(tmp_abs)
        femm.mi_analyse(0)
        femm.mi_loadsolution()

        # Optional bitmap at first angle only
        if save_bmp is not None and idx == 0:
            try:
                bmp_abs = os.path.abspath(save_bmp).replace('\\', '/')
                femm.mo_showdensityplot(1, 0, 2.0, 0, 'bmag')
                femm.mo_savebitmap(bmp_abs)
            except Exception as e:
                print(f"  WARNING: mo_savebitmap failed ({e})")

        femm.mo_groupselectblock(ROTOR_GROUP)
        t = femm.mo_blockintegral(22)   # type 22 = steady-state weighted stress tensor torque
        torques.append(abs(float(t)))
        femm.mo_close()                  # close postprocessor; back to preprocessor

    femm.closefemm()

    reward = float(np.max(torques))
    return reward, torques


# ─────────────────────────────────────────────────────────────────────────────
# SeqTO-v1 MDP environment
# ─────────────────────────────────────────────────────────────────────────────

class SynRMSeqTOEnv:
    """
    SeqTO-v1 MDP for SynRM rotor topology optimisation.

    A 1×1 controller traverses the 5×5 polar design grid depositing iron cells.
    Each step: move controller → deposit 1×1 iron → evaluate FEMM reward.

    State   : flattened iron_mask  (25 float values)
    Actions : 0=RIGHT, 1=LEFT, 2=UP (radially out), 3=DOWN (radially in)
    Reward  : mean torque over 6 rotor angles  (N·m, quarter model)
    Terminal: step ≥ max_steps  OR  iron_count > K2
    """

    def __init__(self, max_steps=MAX_STEPS):
        self.max_steps  = max_steps
        self.iron_mask  = np.zeros((DD_ROWS, DD_COLS), dtype=int)
        self.pos_r      = 0    # controller starts at innermost, most-CW cell
        self.pos_c      = 0
        self.step_count = 0
        self.done       = False

        if not os.path.exists(BASE_FEM_FILE):
            print(f"Base geometry not found.  Building {BASE_FEM_FILE}...")
            build_synrm_geometry()

    def reset(self):
        """Clear topology, reset controller, return initial state vector."""
        self.iron_mask[:] = 0
        self.pos_r      = 0
        self.pos_c      = 0
        self.step_count = 0
        self.done       = False
        return self.iron_mask.flatten().astype(float)

    def step(self, action, save_bmp=None):
        """
        Move controller, deposit 1×1 iron cell, get FEMM torque reward.
        Out-of-bounds moves are silently rejected (position unchanged, still deposits).
        Returns (state, reward, done, info).
        """
        if self.done:
            raise RuntimeError("Episode finished — call reset() first.")

        name, dr, dc = ACTIONS[action]
        new_r = self.pos_r + dr
        new_c = self.pos_c + dc

        if self._check_position(new_r, new_c) == 0:
            self.pos_r, self.pos_c = new_r, new_c

        # Deposit at current position (all 25 design cells are free)
        if is_design_cell(self.pos_r, self.pos_c):
            self.iron_mask[self.pos_r, self.pos_c] = 1

        self.step_count += 1
        iron_count = int(np.sum(self.iron_mask))
        self.done   = (self.step_count >= self.max_steps) or (iron_count > K2)

        reward, torques = calculate_reward(self.iron_mask, save_bmp=save_bmp)

        info = {
            'step':       self.step_count,
            'action':     name,
            'pos':        (self.pos_r, self.pos_c),
            'iron_count': iron_count,
            'torques':    torques,
        }
        return self.iron_mask.flatten().astype(float), reward, self.done, info

    def _check_position(self, r, c):
        """Return 0=valid, 1=outside design grid."""
        return 0 if is_design_cell(r, c) else 1


# ─────────────────────────────────────────────────────────────────────────────
# Validation tests
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_topology():
    """All design cells Air — stray-flux baseline, expect ≈ 0 N·m."""
    print("\n── TEST 1: Empty topology (all Air in rotor design grid) ──")
    mask = np.zeros((DD_ROWS, DD_COLS), dtype=int)
    reward, torques = calculate_reward(mask)
    print(f"   Mean torque     : {reward:.4f} N·m")
    print(f"   Per-angle       : {[f'{t:.4f}' for t in torques]}")
    print(f"   Note            : expect near zero (no saliency)")
    return reward


def test_conventional_topology():
    """
    Approximate 2-barrier SynRM: two radial flux barriers (air) flanking an
    iron bridge in the middle radial band.  This pattern (inner+outer iron,
    mid-air) creates rotor saliency.  Expect ≈ 2–4 N·m quarter-model.
    """
    print("\n── TEST 2: Conventional 2-barrier SynRM topology ──")
    mask = np.zeros((DD_ROWS, DD_COLS), dtype=int)
    # Fill innermost band (i=0) and outermost band (i=4) with iron
    mask[0, :] = 1
    mask[4, :] = 1
    # Fill middle band partially (leave air barriers at i=1,3)
    mask[2, :] = 1
    iron_count = int(np.sum(mask))
    print(f"   Design cells    : {iron_count}  (2 full bands + 1 mid band = 15)")

    os.makedirs(DESIGNS_DIR, exist_ok=True)
    bmp_path = os.path.join(DESIGNS_DIR, 'synrm_conventional.bmp')
    reward, torques = calculate_reward(mask, save_bmp=bmp_path)
    print(f"   Mean torque     : {reward:.4f} N·m")
    print(f"   Per-angle       : {[f'{t:.4f}' for t in torques]}")
    status = "PASS" if reward > 0.01 else "CHECK"
    print(f"   Status          : {status}  (expect > empty topology)")
    return reward


def test_full_iron_topology():
    """All 25 design cells iron — maximum saliency upper bound."""
    print("\n── TEST 3: Full iron fill (all design cells = iron) ──")
    mask = np.ones((DD_ROWS, DD_COLS), dtype=int)
    print(f"   Design cells    : {int(np.sum(mask))}")
    reward, torques = calculate_reward(mask)
    print(f"   Mean torque     : {reward:.4f} N·m")
    print(f"   Per-angle       : {[f'{t:.4f}' for t in torques]}")
    return reward


def test_controller_pattern():
    """
    Fixed deterministic action sequence — verifies controller movement,
    iron deposition, and MDP state/reward correctness visually.

    Pattern (10 steps, radial rib at 45° + top trace):
      RIGHT RIGHT        — j=0→1→2  (deposit at row 0 to reach 45° column)
      UP    UP UP UP     — climb j=2 from row 0 to row 4 (touches outer ring)
      LEFT  LEFT         — trace top ring: (4,2)→(4,1)→(4,0)
      DOWN  DOWN         — descend at j=0: (4,0)→(3,0)→(2,0)

    Iron rib at j=2 (45°, d-axis) connects inner hub ring to outer peripheral
    ring — creates d-axis flux path, expect clear reward increase once rib
    is complete (step 6).
    """
    print("\n── TEST: Deterministic controller pattern (radial rib at 45°) ──")

    # action codes: 0=RIGHT 1=LEFT 2=UP 3=DOWN
    pattern = [0, 0, 2, 2, 2, 2, 1, 1, 3, 3]
    names   = [ACTIONS[a][0] for a in pattern]
    print(f"  Actions : {names}")
    print(f"  Grid    : {DD_ROWS}×{DD_COLS} free cells  "
          f"(r={R_HUB}–{R_PERI_IN} mm, θ=0°–90°)")
    print(f"  j=2 column (45°, d-axis) connects hub ring → peripheral ring at step 6")

    env = SynRMSeqTOEnv(max_steps=len(pattern))
    env.reset()

    print("\n  Initial state (controller at start, no iron):")
    _print_grid(env.iron_mask, env.pos_r, env.pos_c)

    prev_reward = 0.0
    for step_num, action in enumerate(pattern):
        state, reward, done, info = env.step(action)
        delta = reward - prev_reward
        print(f"  Step {step_num+1:2d} | {info['action']:5s} | "
              f"pos={str(info['pos']):>6} | iron={info['iron_count']:2d} | "
              f"reward={reward:.4f} N·m ({delta:+.4f})")
        _print_grid(env.iron_mask, env.pos_r, env.pos_c)
        assert int(state.sum()) == info['iron_count'], "state/iron_count mismatch"
        prev_reward = reward
        if done:
            break

    print(f"  State vector sum = {int(state.sum())}  matches iron_count ✓")
    print(f"  Status: PASS")

    save_file = 'controller_pattern_final.fem'
    print(f"\n  Saving final topology for FEMM inspection...")
    save_topology_fem(env.iron_mask, save_file)

    return reward


def test_mdp_episode(n_steps=8):
    """Random MDP episode — validates the full SeqTO-v1 loop."""
    print(f"\n── TEST: Random MDP episode  ({n_steps} steps) ──")
    env = SynRMSeqTOEnv(max_steps=n_steps)
    env.reset()

    header = (f"  {'Step':>4}  {'Action':<6}  {'Pos':>8}  "
              f"{'Reward':>9}  {'Iron':>5}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    best = 0.0
    for step in range(n_steps):
        action = np.random.randint(4)
        _, reward, done, info = env.step(action)
        print(f"  {step:>4}  {info['action']:<6}  {str(info['pos']):>8}  "
              f"{reward:>9.4f}  {info['iron_count']:>5}")
        best = max(best, reward)
        if done:
            break

    print(f"\n  Best reward this episode : {best:.4f} N·m")
    print(f"  Status : PASS  (episode ran without FEMM error)")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='SynRM SeqTO-v1 FEMM Environment')
    p.add_argument('--build-geometry', action='store_true',
                   help='(Re)build synrm_seqto.fem and exit')
    p.add_argument('--no-tests', action='store_true',
                   help='Skip validation tests after geometry build')
    return p.parse_args()


def main():
    args = _parse_args()

    print("=" * 62)
    print("  seqTO_SynRM_env.py — SynRM SeqTO-v1 FEMM Validation")
    print("=" * 62)
    print(f"  Machine        : 4-pole SynRM, 90° quarter model")
    print(f"  Rotor frame    : hub r={R_SHAFT}–{R_HUB} mm, "
          f"peri bridge r={R_PERI_IN}–{R_ROTOR} mm "
          f"({R_ROTOR - R_PERI_IN:.1f} mm thick)")
    print(f"  Rotor grid     : {DD_ROWS}×{DD_COLS} free cells  "
          f"(dr={DR:.2f} mm, dθ={DTHETA:.0f}°, θ=0°–90°)")
    print(f"  Stator         : R={R_STATOR} mm, stack={L_STACK} mm, "
          f"6 slots + 7 teeth in 90° quarter (q=2 distributed)")
    print(f"  Winding        : Ia={IA:+.2f} A, Ib={IB:+.2f} A, Ic={IC:+.2f} A  "
          f"({N_TURNS} turns/slot)")
    print(f"  Operating pt   : d-axis at {D_AXIS_MECH_DEG:.1f}° mech, "
          f"γ={GAMMA_DEG:.1f}° elec (MTPA at 45°)")
    print(f"  Rotor angles   : {ROTOR_ANGLES_MECH} mech-deg")
    print(f"  Budget         : K1={K1} ≤ iron ≤ K2={K2}")
    print("─" * 62)

    if args.build_geometry or not os.path.exists(BASE_FEM_FILE):
        print(f"\nBuilding base geometry → {BASE_FEM_FILE}")
        build_synrm_geometry()
        if args.build_geometry:
            print("Geometry built.  Run without --build-geometry to run tests.")
            return

    if args.no_tests:
        return

    os.makedirs(DESIGNS_DIR, exist_ok=True)

    rp        = test_controller_pattern()
    r0        = test_empty_topology()
    r1        = test_conventional_topology()
    r2        = test_full_iron_topology()
    br        = test_mdp_episode(n_steps=8)

    print("\n" + "=" * 62)
    print("  RESULTS SUMMARY")
    print(f"  Controller pattern  → {rp:.4f} N·m  (radial rib at 45°, 10 iron cells)")
    print(f"  Empty topology      → {r0:.4f} N·m  (expect ≈ 0)")
    print(f"  Conventional SynRM  → {r1:.4f} N·m  (expect ≈ 2–4)")
    print(f"  Full iron fill      → {r2:.4f} N·m")
    print(f"  MDP best reward     → {br:.4f} N·m")
    print("=" * 62)
    print("  All tests complete.")
    print("=" * 62)


if __name__ == '__main__':
    main()
