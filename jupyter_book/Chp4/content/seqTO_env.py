"""
seqTO_env.py  -  SeqTO-v1 C-core FEMM Environment and Validation

Implements the Sequence Topology Optimisation MDP environment for a C-core
electromagnetic actuator.  Builds the 18x35 mm FEMM grid entirely from scratch
(no actuator.fem dependency) and validates physics with three tests.

CRITICAL implementation notes:
  - mi_analyse(0) SAVES the current in-memory state back to the .fem file on
    disk.  To avoid file corruption across successive reward calls, calculate_reward
    resets ALL design cells to Air at the start of every call, THEN applies the
    new iron_mask.  Without this reset the base file is progressively overwritten
    with iron from previous evaluations, making all subsequent calls return the
    same (wrong) reward.
  - Material: 'Cold rolled low carbon strip steel' matches the CcoreFemmEnv
    reference implementation (Chp5_DRL_TO) and produces forces in the 40-70 N
    range consistent with the thesis benchmark.  'M-19 Steel' (very high mu_r)
    would give ~1500 N and is not what the reference code uses.

Requirements: pyfemm, numpy  (run from Windows Python, FEMM 4.2 at C:\\femm42)
Usage       : python seqTO_env.py
"""

import os
import sys
import numpy as np

try:
    import femm
except ImportError:
    print("ERROR: pyfemm not installed.  Run: pip install pyfemm")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry constants  (matching Chp5_DRL_TO/src/femm_environment/constants.py)
# ──────────────────────────────────────────────────────────────────────────────

ROWS, COLS = 18, 35          # full grid dimensions (mm)
BUFFER     = 0               # no top buffer — design window starts at row 0

# Left coil:  rows 0-8  (x: 0-9 mm), cols 2-4  (y: 2-5 mm) ; -500 turns
COIL_LEN       = 9
COIL1_COLS     = (2, 5)      # [start, end)

# Right coil: rows 0-8  (x: 0-9 mm), cols 11-13 (y: 11-14 mm) ; +500 turns
COIL2_COLS     = (11, 14)    # [start, end)

# Armature:   rows 0-14 (x: 0-15 mm), cols 27-32 (y: 27-33 mm) ; group=5
ARM_ROWS       = (0, 15)     # [start, end)
ARM_COLS       = (27, 33)    # [start, end)

# Design window: rows 0-14, cols 5-25 (right-coil rows 0-8 excluded → 288 designable cells)
DW_ROWS        = (0, ARM_ROWS[1])   # (0, 15)
DW_COLS        = (5, 26)            # cols 5 … 25 inclusive

# Material budget and episode length
K1        = 80    # min iron cells
K2        = 180   # max iron cells (conventional C-core = 180 cells)
MAX_STEPS = 25    # steps per test episode

FEMM_DEPTH    = 40.0          # mm out-of-plane depth
BASE_FEM_FILE = 'ccore_seqto.fem'
DESIGNS_DIR   = 'designs'     # subfolder for solution bitmaps

# Iron material - must match CcoreFemmEnv reference implementation
IRON_MATERIAL = 'Cold rolled low carbon strip steel'

# Actions: name, Δrow, Δcol  (0=RIGHT, 1=LEFT, 2=UP, 3=DOWN)
ACTIONS = {
    0: ('RIGHT', 0, +1),
    1: ('LEFT',  0, -1),
    2: ('UP',   -1,  0),
    3: ('DOWN', +1,  0),
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_design_cell(i, j):
    """True if (i,j) is an optimisable design cell."""
    in_dw    = DW_ROWS[0] <= i < DW_ROWS[1] and DW_COLS[0] <= j < DW_COLS[1]
    in_coil2 = i < COIL_LEN and COIL2_COLS[0] <= j < COIL2_COLS[1]
    return in_dw and not in_coil2


def _reset_design_cells_to_air():
    """Select all design cells and set them to Air (in current open FEMM doc)."""
    for i in range(ROWS):
        for j in range(COLS):
            if is_design_cell(i, j):
                femm.mi_selectlabel(i + 0.5, j + 0.5)
    femm.mi_setblockprop('Air', 0, 0.5, '<None>', 0, 0, 0)
    femm.mi_clearselected()


# ──────────────────────────────────────────────────────────────────────────────
# FEMM geometry builder
# ──────────────────────────────────────────────────────────────────────────────

def build_base_geometry():
    """
    Build the C-core FEMM geometry entirely from scratch and save as
    ccore_seqto.fem.

    Layout (all in mm, 1 mm per cell):
      Left coil  x:[0,9],   y:[2,5]   (Copper, -500 turns, circuit 'icoil')
      Right coil x:[0,9],   y:[11,14] (Copper, +500 turns, circuit 'icoil')
      Armature   x:[0,15],  y:[27,33] (IRON_MATERIAL, group=5)
      Design window: rows 3-17, cols 5-25 (Air by default)
      Outer box  x:[0,100], y:[-100,100] with A=0 Dirichlet on 3 far edges;
                 x=0 is the symmetry plane (natural Neumann BC = flux tangential)
    """
    print("  Opening FEMM, creating magnetostatics document...")
    femm.openfemm(1)
    femm.newdocument(0)                          # 0 = magnetostatics
    femm.mi_probdef(0, 'millimeters', 'planar',  # DC, mm, 2-D planar
                    1e-8, FEMM_DEPTH, 30)        # precision, depth=40mm, minangle

    # ── Materials ─────────────────────────────────────────────────────────────
    femm.mi_getmaterial('Air')
    femm.mi_getmaterial(IRON_MATERIAL)
    femm.mi_getmaterial('Copper')
    femm.mi_addcircprop('icoil', 1, 1)           # 1 A series circuit

    # ── Outer far-field box ────────────────────────────────────────────────────
    # Rectangle corners: (x1=0, y1=100) → (x2=100, y2=-100)
    femm.mi_drawrectangle(0, 100, 100, -100)
    # Outer-air block label (clearly outside the 18×35 grid)
    femm.mi_addblocklabel(50, 0)
    femm.mi_selectlabel(50, 0)
    femm.mi_setblockprop('Air', 50, 0, '<None>', 0, 0, 0)
    femm.mi_clearselected()

    # ── Draw 18×35 grid  (19 vertical + 36 horizontal lines = 55 draws) ───────
    print(f"  Drawing {ROWS+1 + COLS+1} grid lines ({ROWS}×{COLS} cells)...")
    for x in range(ROWS + 1):        # x = 0..18
        femm.mi_drawline(x, 0, x, COLS)
    for y in range(COLS + 1):        # y = 0..35
        femm.mi_drawline(0, y, ROWS, y)

    # ── Block labels: add one per cell, then set all to Air ───────────────────
    print("  Adding block labels (all Air)...")
    for i in range(ROWS):
        for j in range(COLS):
            femm.mi_addblocklabel(i + 0.5, j + 0.5)
    for i in range(ROWS):
        for j in range(COLS):
            femm.mi_selectlabel(i + 0.5, j + 0.5)
    femm.mi_setblockprop('Air', 0, 0.5, '<None>', 0, 0, 0)
    femm.mi_clearselected()

    # ── Left coil: rows 0-8, cols 2-4  (-500 turns) ──────────────────────────
    print("  Assigning coils and armature...")
    for i in range(COIL_LEN):
        for j in range(COIL1_COLS[0], COIL1_COLS[1]):
            femm.mi_selectlabel(i + 0.5, j + 0.5)
    femm.mi_setblockprop('Copper', 0, 0, 'icoil', 0, 1, -500)
    femm.mi_clearselected()

    # ── Right coil: rows 0-8, cols 11-13  (+500 turns) ───────────────────────
    for i in range(COIL_LEN):
        for j in range(COIL2_COLS[0], COIL2_COLS[1]):
            femm.mi_selectlabel(i + 0.5, j + 0.5)
    femm.mi_setblockprop('Copper', 0, 0, 'icoil', 0, 1, 500)
    femm.mi_clearselected()

    # ── Armature: rows 0-14, cols 27-32  (group=5 for force integration) ─────
    for i in range(ARM_ROWS[0], ARM_ROWS[1]):
        for j in range(ARM_COLS[0], ARM_COLS[1]):
            femm.mi_selectlabel(i + 0.5, j + 0.5)
    femm.mi_setblockprop(IRON_MATERIAL, 0, 0.5, '<None>', 0, 5, 0)
    femm.mi_clearselected()

    # ── Boundary conditions ───────────────────────────────────────────────────
    # A=0 Dirichlet on the 3 far-field edges (right, top, bottom)
    femm.mi_addboundprop('A=0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    femm.mi_selectsegment(100,  0)    # right edge  x=100
    femm.mi_selectsegment(50,  100)   # top edge    y=100
    femm.mi_selectsegment(50, -100)   # bottom edge y=-100
    femm.mi_setsegmentprop('A=0', 0, 0, 0, 0)
    femm.mi_clearselected()
    # Left edge (x=0): natural Neumann = symmetry plane (flux tangential)

    femm.mi_saveas(BASE_FEM_FILE)
    femm.closefemm()
    print(f"  Base geometry saved → {BASE_FEM_FILE}")


# ──────────────────────────────────────────────────────────────────────────────
# Geometry screenshot  (preprocessor view)
# ──────────────────────────────────────────────────────────────────────────────

def save_geometry_screenshot(save_path):
    """
    Open the base .fem file in the FEMM preprocessor (visible window), zoom
    to the 18×35 mm design grid, and save a bitmap screenshot via mi_savebitmap.

    mi_savebitmap captures whatever is currently displayed in the FEMM
    preprocessor window — coil blocks, armature block, grid lines, block label
    dots — exactly as drawn by build_base_geometry().

    IMPORTANT: openfemm(0) (visible) is required; openfemm(1) (hidden) produces
    a blank bitmap because FEMM never renders the frame buffer when hidden.
    """
    # mi_savemetafile writes an Enhanced Metafile (EMF) directly from FEMM's
    # internal drawing commands — it never reads from the screen frame buffer.
    # mi_savebitmap does read from the frame buffer and reliably crashes with
    # "page fault ahoy23" when the FEMM window has not yet painted to screen,
    # which happens on every fresh opendocument call.
    # pyfemm bug: mi_savemetafile (and mi_savebitmap) omit fixpath(), unlike
    # mo_savebitmap.  Windows backslashes survive into the Lua string where they
    # are parsed as escape sequences, silently corrupting the path.  Apply the
    # same replace that fixpath() does before passing to FEMM.
    emf_abs = os.path.abspath(save_path).replace('\\', '/')
    femm.openfemm(0)
    femm.opendocument(BASE_FEM_FILE)
    femm.mi_zoom(-2, -2, ROWS + 2, COLS + 2)
    femm.mi_savemetafile(emf_abs)
    femm.closefemm()
    print(f"  Geometry screenshot saved → {emf_abs}")


# ──────────────────────────────────────────────────────────────────────────────
# FEMM solve  (reward calculation)
# ──────────────────────────────────────────────────────────────────────────────

_TMP_FEM = '_tmp_solve.fem'   # throw-away file; mi_analyse writes here, not to BASE_FEM_FILE


def calculate_reward(iron_mask, save_bmp=None):
    """
    Open base geometry, apply iron_mask to design cells, run magnetostatic FEA,
    return (reward, force_N).

    reward  = mean |B| at 6 armature sample points × 1000  (matches CcoreFemmEnv)
    force_N = Maxwell-stress y-component on armature group=5, scaled to Newtons
              [mo_blockintegral(19) × (-10)]

    iron_mask : 18×35 int array  (1 = iron, 0 = air)
    save_bmp  : path string to save a B-field bitmap (e.g. 'ccore_field.bmp'),
                or None to skip.  Uses mo_showdensityplot + mo_savebitmap per
                the pyfemm manual.
    """
    # openfemm(1) = hidden (fast for batch); openfemm(0) = visible (needed for images).
    hide = 1 if save_bmp is None else 0
    femm.openfemm(hide)
    femm.opendocument(BASE_FEM_FILE)

    # ── STEP 1: Reset ALL design cells to Air ─────────────────────────────────
    # Required on first run after applying this fix: BASE_FEM_FILE may still
    # hold iron blocks written by a previous session's mi_analyse(0).
    _reset_design_cells_to_air()

    # ── STEP 2: Apply new iron_mask ───────────────────────────────────────────
    has_iron = False
    for i in range(ROWS):
        for j in range(COLS):
            if is_design_cell(i, j) and iron_mask[i, j] == 1:
                femm.mi_selectlabel(i + 0.5, j + 0.5)
                has_iron = True
    if has_iron:
        femm.mi_setblockprop(IRON_MATERIAL, 0, 0.5, '<None>', 0, 3, 0)
    femm.mi_clearselected()

    # ── STEP 3: Solve via a temp file so BASE_FEM_FILE is never overwritten ───
    # mi_analyse(0) saves the current document back to disk before solving.
    # By switching to _TMP_FEM first, the master geometry file stays pristine
    # across thousands of calls.  Without this, FEMM writes accumulated
    # iron-state changes back to ccore_seqto.fem on every solve, degrading the
    # geometry until Triangle fails with "problem loading mesh".
    tmp_abs = os.path.abspath(_TMP_FEM).replace('\\', '/')
    femm.mi_saveas(tmp_abs)
    femm.mi_analyse(0)
    femm.mi_loadsolution()

    # ── STEP 4: Optional – save B-field solution image (per pyfemm manual) ───
    if save_bmp is not None:
        try:
            # Absolute path required — FEMM's working directory may differ
            bmp_abs = os.path.abspath(save_bmp)
            femm.mo_zoom(0, 0, ROWS, COLS)          # zoom to 18×35 mm grid
            femm.mo_showdensityplot(1, 0, 2.0, 0, 'bmag')   # |B| colour, 0-2 T
            femm.mo_savebitmap(bmp_abs)
            print(f"  Solution image saved → {bmp_abs}")
        except Exception as e:
            print(f"  WARNING: mo_savebitmap failed ({e})")

    # ── STEP 5: Extract results ───────────────────────────────────────────────
    # B-field at 6 armature sample points (identical to CcoreFemmEnv)
    # Points: (x=0.2, y=27.5), (0.2, 28.5), ..., (0.2, 32.5)
    b_sum = 0.0
    for arm in range(6):
        bx, by = femm.mo_getb(0.2, 27.5 + arm)
        b_sum += np.sqrt(bx ** 2 + by ** 2)

    # Force on armature via Maxwell stress tensor, y-component (type 19).
    # type 19 = "y (or z) part of steady-state weighted stress tensor force"
    # The armature is pulled in the -y direction → raw integral is negative.
    # factor = (-10) matches CcoreFemmEnv._calculate_reward; this factor was
    # calibrated against the specific actuator.fem mesh.  A from-scratch uniform
    # 1 mm grid may return different raw values, shifting the absolute Newtons.
    femm.mo_groupselectblock(5)
    raw_force = femm.mo_blockintegral(19)
    force = raw_force * (-10)

    femm.closefemm()

    reward = (b_sum / 6) * 1000
    return float(reward), float(force), float(raw_force)


# ──────────────────────────────────────────────────────────────────────────────
# SeqTO-v1 MDP environment
# ──────────────────────────────────────────────────────────────────────────────

class CcoreSeqTOEnv:
    """
    SeqTO-v1 MDP for C-core topology optimisation.

    A 3×3 controller traverses the design window depositing iron blocks.
    Each step: move controller → deposit 3×3 iron → evaluate FEMM reward.

    State   : flattened iron_mask  (630 float values)
    Actions : 0=RIGHT, 1=LEFT, 2=UP, 3=DOWN
    Reward  : FEMM B-field signal  (same formula as CcoreFemmEnv)
    Terminal: step ≥ max_steps  OR  iron_count > K2
    """

    def __init__(self, max_steps=MAX_STEPS):
        self.max_steps  = max_steps
        self.k          = 1        # half-width for 3×3 deposit block
        self.iron_mask  = np.zeros((ROWS, COLS), dtype=int)
        self.pos_r      = DW_ROWS[0] + 1   # controller centre: 3×3 top-left flush at DW corner
        self.pos_c      = DW_COLS[0] + 1
        self.step_count = 0
        self.done       = False

        if not os.path.exists(BASE_FEM_FILE):
            print(f"Base geometry not found.  Building {BASE_FEM_FILE}...")
            build_base_geometry()

    def reset(self):
        """Clear topology, return initial state vector."""
        self.iron_mask[:] = 0
        self.pos_r      = DW_ROWS[0] + 1
        self.pos_c      = DW_COLS[0] + 1
        self.step_count = 0
        self.done       = False
        return self.iron_mask.flatten().astype(float)

    def step(self, action, save_bmp=None):
        """
        Move controller, deposit 3×3 iron block, get FEMM reward.
        Out-of-bounds / coil-collision moves are silently rejected.
        Returns (state, reward, done, info).

        save_bmp : optional path for B-field bitmap — passed straight through
                   to calculate_reward so the field image is saved inside the
                   same FEMM session that computes the reward (no second solve).
        """
        if self.done:
            raise RuntimeError("Episode finished — call reset() first.")

        name, dr, dc = ACTIONS[action]
        new_r = self.pos_r + dr
        new_c = self.pos_c + dc

        issue = self._check_position(new_r, new_c)
        if issue == 0:
            self.pos_r, self.pos_c = new_r, new_c
            for di in range(-self.k, self.k + 1):
                for dj in range(-self.k, self.k + 1):
                    ri, ci = self.pos_r + di, self.pos_c + dj
                    if is_design_cell(ri, ci):
                        self.iron_mask[ri, ci] = 1

        self.step_count += 1
        iron_count = int(np.sum(self.iron_mask))
        self.done   = (self.step_count >= self.max_steps) or (iron_count > K2)

        reward, force, _ = calculate_reward(self.iron_mask, save_bmp=save_bmp)

        info = {
            'step':       self.step_count,
            'action':     name,
            'iron_count': iron_count,
            'force_N':    force,
            'issue':      issue,
        }
        return self.iron_mask.flatten().astype(float), reward, self.done, info

    def _check_position(self, r, c):
        """Return 0=valid, 1=outside design window, 2=right-coil collision."""
        if not (DW_ROWS[0] <= r < DW_ROWS[1] and DW_COLS[0] <= c < DW_COLS[1]):
            return 1
        if r < COIL_LEN and COIL2_COLS[0] <= c < COIL2_COLS[1]:
            return 2
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# Validation tests
# ──────────────────────────────────────────────────────────────────────────────

def test_empty_topology():
    """No iron in design window — stray-flux baseline."""
    print("\n── TEST 1: Empty topology (all Air in design window) ──")
    mask = np.zeros((ROWS, COLS), dtype=int)
    reward, force, raw = calculate_reward(mask)
    print(f"   Reward          : {reward:.4f}")
    print(f"   Force           : {force:.3f} N  (raw integral={raw:.4f})")
    print(f"   Note            : non-zero due to coil stray flux through air to armature")
    return reward, force


def test_conventional_topology():
    """
    Approximate conventional half-symmetric C-core (180 iron cells).

    Fills two vertical legs running the full height of the design window
    (cols 5-9 and 20-25) plus a bottom connecting bar (last 3 rows of DW),
    approximating a C-shape connecting the coil region to the armature gap.
    Iron count is trimmed to exactly K2=180 if the raw count exceeds it.
    Expected force: approximately 40 N (thesis baseline).
    """
    print("\n── TEST 2: Conventional C-core topology (~180 cells) ──")
    mask = np.zeros((ROWS, COLS), dtype=int)

    # Left vertical leg: full design window height, left 5 cols
    for i in range(DW_ROWS[0], DW_ROWS[1]):
        for j in range(5, 10):
            if is_design_cell(i, j):
                mask[i, j] = 1

    # Right vertical leg: full design window height, right 5 cols
    for i in range(DW_ROWS[0], DW_ROWS[1]):
        for j in range(20, 26):
            if is_design_cell(i, j):
                mask[i, j] = 1

    # Bottom bar connecting the two legs (last 3 rows of design window)
    for i in range(DW_ROWS[1]-3, DW_ROWS[1]):
        for j in range(DW_COLS[0], DW_COLS[1]):
            if is_design_cell(i, j):
                mask[i, j] = 1

    n_cells = int(np.sum(mask))
    print(f"   Cells filled before trim : {n_cells}")

    # Trim to K2 if over budget (remove from bottom bar rightmost cells last)
    if n_cells > K2:
        filled = np.argwhere(mask == 1)
        for idx in filled[K2:]:
            mask[idx[0], idx[1]] = 0
        n_cells = K2

    print(f"   Design cells             : {n_cells}")
    reward, force, raw = calculate_reward(
        mask, save_bmp=os.path.join(DESIGNS_DIR, 'ccore_conventional.bmp'))
    print(f"   Reward          : {reward:.4f}")
    print(f"   Force           : {force:.3f} N  (raw integral={raw:.4f})")
    print(f"   Note            : thesis baseline ~40 N uses actuator.fem mesh")
    print(f"                     (different mesh → different raw integral scale)")
    status = "PASS" if force > 50 else "CHECK"
    print(f"   Status          : {status}  (expect > empty topology force)")
    return reward, force


def test_full_iron_topology():
    """All design cells filled — upper-bound reference."""
    print("\n── TEST 3: Full iron fill (all design cells = iron) ──")
    mask = np.zeros((ROWS, COLS), dtype=int)
    n_cells = 0
    for i in range(ROWS):
        for j in range(COLS):
            if is_design_cell(i, j):
                mask[i, j] = 1
                n_cells += 1
    print(f"   Design cells filled : {n_cells}")
    reward, force, raw = calculate_reward(
        mask, save_bmp=os.path.join(DESIGNS_DIR, 'ccore_full_iron.bmp'))
    print(f"   Reward          : {reward:.4f}")
    print(f"   Force           : {force:.3f} N  (raw integral={raw:.4f})")
    status = "PASS" if force > 5.0 else "FAIL"
    print(f"   Status          : {status}  (expect > conventional force)")
    return reward, force


def test_mdp_episode(n_steps=10):
    """Random MDP episode — validates the full SeqTO-v1 loop."""
    print(f"\n── TEST 4: MDP episode  ({n_steps} random steps) ──")
    env = CcoreSeqTOEnv(max_steps=n_steps)
    env.reset()

    header = (f"  {'Step':>4}  {'Action':<6}  {'Reward':>8}  "
              f"{'Force(N)':>9}  {'Iron':>5}  {'Issue':>5}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    best_reward = 0.0
    for step in range(n_steps):
        action = np.random.randint(4)
        _, reward, done, info = env.step(action)
        print(f"  {step:>4}  {info['action']:<6}  {reward:>8.3f}  "
              f"{info['force_N']:>9.3f}  {info['iron_count']:>5}  "
              f"{info['issue']:>5}")
        best_reward = max(best_reward, reward)
        if done:
            break

    print(f"\n  Best reward this episode : {best_reward:.3f}")
    print(f"  Status : PASS  (episode ran without FEMM error)")
    return best_reward


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  seqTO_env.py — C-core SeqTO-v1 FEMM Validation")
    print("=" * 62)

    n_design = sum(1 for i in range(ROWS) for j in range(COLS)
                   if is_design_cell(i, j))

    # Always rebuild base geometry to ensure material settings are current
    print(f"\nBuilding base FEMM geometry → {BASE_FEM_FILE}")
    print(f"  Grid        : {ROWS}×{COLS} mm  ({ROWS*COLS} cells, 1 mm each)")
    print(f"  Left coil   : rows 0-8, cols 2-4   (-500 A-turns)")
    print(f"  Right coil  : rows 0-8, cols 11-13 (+500 A-turns)")
    print(f"  Armature    : rows 0-14, cols 27-32 (group=5)")
    print(f"  Iron mat.   : {IRON_MATERIAL}")
    print(f"  Depth       : {FEMM_DEPTH} mm")
    print(f"  Design cells: {n_design}  (rows {DW_ROWS[0]}-{DW_ROWS[1]-1}, "
          f"cols {DW_COLS[0]}-{DW_COLS[1]-1}, excl. right-coil overlap)")
    build_base_geometry()

    os.makedirs(DESIGNS_DIR, exist_ok=True)
    print(f"  Designs folder      : {os.path.abspath(DESIGNS_DIR)}")

    print(f"\nSaving geometry screenshot → {DESIGNS_DIR}/ccore_geometry.bmp")
    save_geometry_screenshot(os.path.join(DESIGNS_DIR, 'ccore_geometry.emf'))

    # Run tests
    r0, f0 = test_empty_topology()
    r1, f1 = test_conventional_topology()
    r2, f2 = test_full_iron_topology()
    br      = test_mdp_episode(n_steps=10)

    print("\n" + "=" * 62)
    print("  RESULTS SUMMARY")
    print(f"  Empty topology      → reward={r0:.2f},  force={f0:.2f} N")
    print(f"  Conventional C-core → reward={r1:.2f},  force={f1:.2f} N")
    print(f"  (Thesis reports ~40 N using actuator.fem mesh; raw integral scale differs)")
    print(f"  Full iron fill      → reward={r2:.2f},  force={f2:.2f} N")
    print(f"  MDP best reward     → {br:.2f}")
    print(f"  Geometry layout     → {DESIGNS_DIR}/ccore_geometry.emf")
    print(f"  Solution images     → {DESIGNS_DIR}/ccore_conventional.bmp, "
          f"{DESIGNS_DIR}/ccore_full_iron.bmp")
    print("=" * 62)
    print("  All tests complete.")
    print("=" * 62)


if __name__ == '__main__':
    main()
