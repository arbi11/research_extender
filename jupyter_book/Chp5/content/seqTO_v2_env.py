"""
seqTO_v2_env.py -- SeqTO-v2 environment for Chp5 DRL on the C-core actuator.

Flavour A extension of Chp4's CcoreSeqTOEnv:

  PRESERVES from Chp4
  -------------------
  * 4-action deposition controller (RIGHT/LEFT/UP/DOWN)
  * 3x3 brush, in_coil2 exclusion -> connectivity-by-construction guarantee
  * FEMM-based reward (mean |B| at 6 armature points x 1000) AND force in N
  * Same geometry constants (18x35 grid, K1=80, K2=180, MAX_STEPS=34)

  UPGRADES for DRL
  ----------------
  * State: 3-channel [material, flux, boundary] tensor for CNN policies
  * Flux channel: per-cell |B| sampled from FEMM (every step by default)
  * Reward: force in Newtons (DRL target) OR mean |B| x 1000 (Chp4 compat)
  * Gym-style API: reset() -> obs; step(a) -> (obs, r, done, info)

State (observation): np.ndarray, shape (3, 18, 35), float32
    Channel 0  material distribution     (0=air, 1=iron)
    Channel 1  flux density |B|          (sampled from FEMM, [0, 1] after norm)
    Channel 2  boundary / fixed mask     (precomputed; 1=coil or armature, 0=else)

Action: int in {0, 1, 2, 3}  -- RIGHT/LEFT/UP/DOWN

Reward (configurable via constructor):
    reward_kind='force'  (default)  : Maxwell-stress y-force in Newtons
    reward_kind='mean_b'            : mean |B| at 6 armature points x 1000

Termination: step_count >= max_steps OR iron_count > K2.

FEMM dependency
---------------
Requires pyfemm + FEMM 4.2 (Windows).  Flux sampling adds ROWS*COLS=630
mo_getb calls per step inside the SAME FEMM session as the solve (no second
openfemm overhead).  Pure-Python fallback (random flux noise) when pyfemm is
absent -- for shape/dtype/plumbing tests in WSL without FEMM installed.

Geometry constants are duplicated from Chp4/content/seqTO_env.py rather than
cross-imported, so Chp5 stays self-contained.  Any future change to the C-core
geometry should be mirrored here manually.

Setup
-----
Self-contained -- Chp5 has no cross-chapter file dependency on Chp4.
The env reads its base FEMM geometry (default ``ccore_seqto.fem``) at run
time; if missing it is BUILT locally on first instantiation via
``build_base_geometry()``.  Geometry constants and the build routine are
duplicated verbatim from Chp4/seqTO_env.py.

Parallel runs
-------------
DQN and A2C can be launched concurrently by giving each script its own
``base_fem_file`` and ``tmp_fem`` so they don't share files on disk:

    python dqn_seqto.py    # default: ccore_seqto_dqn.fem + _tmp_solve_v2_dqn.fem
    python a2c_seqto.py    # default: ccore_seqto_a2c.fem + _tmp_solve_v2_a2c.fem

Each script auto-builds its own ``.fem`` copy on first run.
"""

import os
import numpy as np

try:
    import femm
    _FEMM_AVAILABLE = True
except ImportError:
    _FEMM_AVAILABLE = False


# ── Geometry constants (mirror of Chp4/seqTO_env.py) ───────────────────────
ROWS, COLS    = 18, 35
COIL_LEN      = 9
COIL1_COLS    = (2, 5)
COIL2_COLS    = (11, 14)
ARM_ROWS      = (0, 15)
ARM_COLS      = (27, 33)
DW_ROWS       = (0, 15)
DW_COLS       = (5, 26)
K1            = 80
K2            = 180
MAX_STEPS_DEFAULT = 34
FEMM_DEPTH    = 40.0
IRON_MATERIAL = 'Cold rolled low carbon strip steel'

ACTIONS = {
    0: ('RIGHT',  0, +1),
    1: ('LEFT',   0, -1),
    2: ('UP',    -1,  0),
    3: ('DOWN',  +1,  0),
}


def is_design_cell(i, j):
    """True if (i, j) is a designable cell (inside DW, not in COIL2 shadow)."""
    in_dw    = DW_ROWS[0] <= i < DW_ROWS[1] and DW_COLS[0] <= j < DW_COLS[1]
    in_coil2 = i < COIL_LEN and COIL2_COLS[0] <= j < COIL2_COLS[1]
    return in_dw and not in_coil2


def build_boundary_mask():
    """Channel-2 mask: 1 at structural cells (coils + armature), 0 elsewhere.

    Constant across the episode -- precomputed once and reused.  Tells the
    CNN where iron CANNOT be placed by the controller (these regions are
    fixed features of the geometry).
    """
    m = np.zeros((ROWS, COLS), dtype=np.float32)
    for i in range(ROWS):
        for j in range(COLS):
            if i < COIL_LEN and (COIL1_COLS[0] <= j < COIL1_COLS[1]
                                 or COIL2_COLS[0] <= j < COIL2_COLS[1]):
                m[i, j] = 1.0
            elif ARM_ROWS[0] <= i < ARM_ROWS[1] and ARM_COLS[0] <= j < ARM_COLS[1]:
                m[i, j] = 1.0
    return m


# ────────────────────────────────────────────────────────────────────────────
# C-core base geometry builder
#
# Duplicated verbatim from Chp4/content/seqTO_env.py::build_base_geometry so
# Chp5 is fully self-contained -- the env builds its own ccore_seqto.fem on
# first run if the file is missing.  Any future change to the C-core
# geometry in Chp4 should be mirrored here manually.
# ────────────────────────────────────────────────────────────────────────────

def build_base_geometry(base_fem_file='ccore_seqto.fem'):
    """Build the C-core FEMM geometry from scratch and save it.

    Layout (all in mm, 1 mm per cell):
      Left coil  x:[0,9],   y:[2,5]   (Copper, -500 turns, circuit 'icoil')
      Right coil x:[0,9],   y:[11,14] (Copper, +500 turns, circuit 'icoil')
      Armature   x:[0,15],  y:[27,33] (IRON_MATERIAL, group=5)
      Design window: rows 3-17, cols 5-25 (Air by default)
      Outer box  x:[0,100], y:[-100,100] with A=0 Dirichlet on 3 far edges
    """
    if not _FEMM_AVAILABLE:
        raise RuntimeError(
            "pyfemm is not installed; cannot build base geometry.  "
            "Run on Windows with FEMM 4.2 + pyfemm.")
    print(f"  [seqTO_v2_env] building base geometry -> {base_fem_file}")
    femm.openfemm(1)
    femm.newdocument(0)
    femm.mi_probdef(0, 'millimeters', 'planar', 1e-8, FEMM_DEPTH, 30)

    femm.mi_getmaterial('Air')
    femm.mi_getmaterial(IRON_MATERIAL)
    femm.mi_getmaterial('Copper')
    femm.mi_addcircprop('icoil', 1, 1)

    femm.mi_drawrectangle(0, 100, 100, -100)
    femm.mi_addblocklabel(50, 0)
    femm.mi_selectlabel(50, 0)
    femm.mi_setblockprop('Air', 50, 0, '<None>', 0, 0, 0)
    femm.mi_clearselected()

    for x in range(ROWS + 1):
        femm.mi_drawline(x, 0, x, COLS)
    for y in range(COLS + 1):
        femm.mi_drawline(0, y, ROWS, y)

    for i in range(ROWS):
        for j in range(COLS):
            femm.mi_addblocklabel(i + 0.5, j + 0.5)
    for i in range(ROWS):
        for j in range(COLS):
            femm.mi_selectlabel(i + 0.5, j + 0.5)
    femm.mi_setblockprop('Air', 0, 0.5, '<None>', 0, 0, 0)
    femm.mi_clearselected()

    # Left coil: rows 0-8, cols 2-4  (-500 turns)
    for i in range(COIL_LEN):
        for j in range(COIL1_COLS[0], COIL1_COLS[1]):
            femm.mi_selectlabel(i + 0.5, j + 0.5)
    femm.mi_setblockprop('Copper', 0, 0, 'icoil', 0, 1, -500)
    femm.mi_clearselected()

    # Right coil: rows 0-8, cols 11-13  (+500 turns)
    for i in range(COIL_LEN):
        for j in range(COIL2_COLS[0], COIL2_COLS[1]):
            femm.mi_selectlabel(i + 0.5, j + 0.5)
    femm.mi_setblockprop('Copper', 0, 0, 'icoil', 0, 1, 500)
    femm.mi_clearselected()

    # Armature: rows 0-14, cols 27-32 (group=5 for force integration)
    for i in range(ARM_ROWS[0], ARM_ROWS[1]):
        for j in range(ARM_COLS[0], ARM_COLS[1]):
            femm.mi_selectlabel(i + 0.5, j + 0.5)
    femm.mi_setblockprop(IRON_MATERIAL, 0, 0.5, '<None>', 0, 5, 0)
    femm.mi_clearselected()

    # Far-field A=0 boundary on 3 outer edges (x=0 = symmetry plane, natural Neumann)
    femm.mi_addboundprop('A=0', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    femm.mi_selectsegment(100,    0)
    femm.mi_selectsegment(50,   100)
    femm.mi_selectsegment(50,  -100)
    femm.mi_setsegmentprop('A=0', 0, 0, 0, 0)
    femm.mi_clearselected()

    femm.mi_saveas(base_fem_file)
    femm.closefemm()
    print(f"  [seqTO_v2_env] base geometry saved")


# ────────────────────────────────────────────────────────────────────────────
# FEMM cycle: apply mask, solve, extract reward / force / per-cell flux grid
# ────────────────────────────────────────────────────────────────────────────

def _solve_and_extract(iron_mask, base_fem_file, tmp_fem, flux_norm=2.0):
    """One FEMM cycle: returns (mean_b_reward, force_N, flux_grid).

    Mirrors Chp4 seqTO_env.calculate_reward() but adds an 18x35 mo_getb sweep
    on the same FEMM session.  No second openfemm/closefemm overhead.
    """
    femm.openfemm(1)
    femm.opendocument(base_fem_file)

    # Reset all design cells to Air (counter prior session's writes)
    for i in range(ROWS):
        for j in range(COLS):
            if is_design_cell(i, j):
                femm.mi_selectlabel(i + 0.5, j + 0.5)
    femm.mi_setblockprop('Air', 0, 0.5, '<None>', 0, 1, 0)
    femm.mi_clearselected()

    # Apply current iron_mask
    has_iron = False
    for i in range(ROWS):
        for j in range(COLS):
            if is_design_cell(i, j) and iron_mask[i, j] == 1:
                femm.mi_selectlabel(i + 0.5, j + 0.5)
                has_iron = True
    if has_iron:
        femm.mi_setblockprop(IRON_MATERIAL, 0, 0.5, '<None>', 0, 3, 0)
    femm.mi_clearselected()

    # Solve via temp file
    tmp_abs = os.path.abspath(tmp_fem).replace('\\', '/')
    femm.mi_saveas(tmp_abs)
    femm.mi_analyse(0)
    femm.mi_loadsolution()

    # Mean |B| at 6 armature sample points (Chp4 reward signal)
    b_sum = 0.0
    for arm in range(6):
        bx, by = femm.mo_getb(0.2, 27.5 + arm)
        b_sum += np.sqrt(bx ** 2 + by ** 2)
    mean_b_reward = (b_sum / 6.0) * 1000.0

    # Maxwell-stress y-force on armature (group 5), already in Newtons
    femm.mo_groupselectblock(5)
    raw_force = femm.mo_blockintegral(19)
    force_N   = raw_force * (-10.0)
    femm.mo_clearblock()

    # Per-cell flux sampling (new for v2)
    flux = np.zeros((ROWS, COLS), dtype=np.float32)
    for i in range(ROWS):
        for j in range(COLS):
            bx, by = femm.mo_getb(i + 0.5, j + 0.5)
            flux[i, j] = float(np.hypot(bx, by))
    flux = np.clip(flux / flux_norm, 0.0, 1.0)

    femm.closefemm()
    return float(mean_b_reward), float(force_N), flux


# ────────────────────────────────────────────────────────────────────────────
# Env class
# ────────────────────────────────────────────────────────────────────────────

class CcoreSeqTOv2Env:
    """SeqTO-v2 C-core environment - Flavour A: Chp4 controller mechanics + CNN-ready state."""

    OBSERVATION_SHAPE = (3, ROWS, COLS)
    ACTION_SIZE       = 4

    def __init__(self,
                 max_steps=MAX_STEPS_DEFAULT,
                 flux_norm=2.0,
                 reward_kind='force',
                 random_start=False,
                 seed=None,
                 base_fem_file='ccore_seqto.fem',
                 tmp_fem='_tmp_solve_v2.fem'):
        if reward_kind not in ('force', 'mean_b'):
            raise ValueError(f"reward_kind must be 'force' or 'mean_b', got {reward_kind!r}")
        self.max_steps     = max_steps
        self.flux_norm     = flux_norm
        self.reward_kind   = reward_kind
        self.random_start  = random_start
        self._rng          = np.random.default_rng(seed)
        self._boundary     = build_boundary_mask()
        self.base_fem_file = base_fem_file
        self.tmp_fem       = tmp_fem

        # Self-contained: build the base .fem locally if it doesn't exist.
        # Callers can use different base_fem_file values to run in parallel
        # with their own copies (e.g. dqn_seqto.py uses ccore_seqto_dqn.fem,
        # a2c_seqto.py uses ccore_seqto_a2c.fem).
        if _FEMM_AVAILABLE and not os.path.exists(self.base_fem_file):
            build_base_geometry(self.base_fem_file)

        # Mutable state
        self.iron_mask  = np.zeros((ROWS, COLS), dtype=np.int32)
        self.flux_mask  = np.zeros((ROWS, COLS), dtype=np.float32)
        self.pos_r      = DW_ROWS[0] + 1
        self.pos_c      = DW_COLS[0] + 1
        self.step_count = 0
        self.done       = False

    # ── Gym-style metadata (lightweight; no gym dependency required) ──────
    @property
    def observation_space(self):
        return {
            'shape': self.OBSERVATION_SHAPE,
            'dtype': 'float32',
            'low':   0.0,
            'high':  1.0,
        }

    @property
    def action_space(self):
        return {'n': self.ACTION_SIZE, 'kind': 'discrete'}

    # ── Helpers ───────────────────────────────────────────────────────────
    def _observation(self):
        return np.stack([
            self.iron_mask.astype(np.float32),
            self.flux_mask,
            self._boundary,
        ], axis=0)

    def _random_design_cell(self):
        """Sample a uniformly random in-design-domain (i, j)."""
        candidates = [(i, j) for i in range(DW_ROWS[0], DW_ROWS[1])
                             for j in range(DW_COLS[0], DW_COLS[1])
                             if is_design_cell(i, j)]
        i, j = candidates[int(self._rng.integers(len(candidates)))]
        return i, j

    # ── Core API ──────────────────────────────────────────────────────────
    def reset(self):
        """Clear topology, reset controller, return initial 3-channel state."""
        self.iron_mask[:] = 0
        self.flux_mask[:] = 0.0
        if self.random_start:
            self.pos_r, self.pos_c = self._random_design_cell()
        else:
            self.pos_r = DW_ROWS[0] + 1
            self.pos_c = DW_COLS[0] + 1
        self.step_count = 0
        self.done       = False
        return self._observation()

    def step(self, action):
        """Move controller, deposit 3x3 iron, solve FEMM, sample flux grid.

        Returns (observation, reward, done, info) per gym convention.
        """
        if self.done:
            raise RuntimeError("Episode finished -- call reset() first.")

        # 1) Controller move (same as Chp4)
        _, dr, dc = ACTIONS[int(action)]
        new_r = self.pos_r + dr
        new_c = self.pos_c + dc
        in_dw    = DW_ROWS[0] <= new_r < DW_ROWS[1] and DW_COLS[0] <= new_c < DW_COLS[1]
        in_coil2 = new_r < COIL_LEN and COIL2_COLS[0] <= new_c < COIL2_COLS[1]
        if in_dw and not in_coil2:
            self.pos_r, self.pos_c = new_r, new_c
            # 3x3 brush deposit at new position (in_coil2 cells excluded by is_design_cell)
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ri, ci = self.pos_r + di, self.pos_c + dj
                    if is_design_cell(ri, ci):
                        self.iron_mask[ri, ci] = 1

        self.step_count += 1
        iron_count = int(np.sum(self.iron_mask))
        self.done = (self.step_count >= self.max_steps) or (iron_count > K2)

        # 2) FEMM solve + per-cell flux sampling (or pure-Python fallback)
        if _FEMM_AVAILABLE:
            mean_b, force_N, flux = _solve_and_extract(
                self.iron_mask, self.base_fem_file, self.tmp_fem, self.flux_norm)
            self.flux_mask = flux
        else:
            mean_b, force_N = 0.0, 0.0
            # Random low-amplitude noise so downstream code sees a non-zero tensor
            self.flux_mask = (self._rng.random((ROWS, COLS), dtype=np.float32)
                              * 0.1).astype(np.float32)

        reward = force_N if self.reward_kind == 'force' else mean_b

        info = {
            'mean_b':     mean_b,
            'force':      force_N,
            'iron_count': iron_count,
            'pos':        (self.pos_r, self.pos_c),
            'step':       self.step_count,
            'femm':       _FEMM_AVAILABLE,
        }
        return self._observation(), reward, self.done, info

    def render(self, mode='human'):
        """Quick text dump of the iron mask (no graphical render yet)."""
        print(f"  step {self.step_count:2d}  pos=({self.pos_r}, {self.pos_c})  "
              f"iron={int(self.iron_mask.sum())}/288  done={self.done}")

    def close(self):
        """No persistent resources to release (FEMM is opened per-step)."""
        pass


# ────────────────────────────────────────────────────────────────────────────
# Smoke test (run as `python seqTO_v2_env.py`)
# ────────────────────────────────────────────────────────────────────────────

def _smoke_test():
    """Verify shapes, dtypes, controller mechanics without FEMM."""
    print("=" * 60)
    print(f"  seqTO_v2_env smoke test  (FEMM available: {_FEMM_AVAILABLE})")
    print("=" * 60)

    env = CcoreSeqTOv2Env(max_steps=5, seed=42)
    obs = env.reset()
    print(f"reset()      : obs.shape={obs.shape}  dtype={obs.dtype}  "
          f"min={obs.min():.3f}  max={obs.max():.3f}")
    assert obs.shape == (3, ROWS, COLS), f"bad shape: {obs.shape}"
    assert obs.dtype == np.float32, f"bad dtype: {obs.dtype}"
    assert (obs[0] == 0).all(), 'iron channel should be 0 at reset'
    assert (obs[2] == env._boundary).all(), 'boundary channel mismatch'

    # Take 5 deterministic steps (RIGHT, RIGHT, DOWN, DOWN, LEFT)
    for t, a in enumerate([0, 0, 3, 3, 1]):
        obs, r, done, info = env.step(a)
        print(f"step {t+1}: action={ACTIONS[a][0]:5s}  obs.shape={obs.shape}  "
              f"reward={r:.4f}  iron={info['iron_count']:2d}  done={done}  "
              f"pos={info['pos']}")
        assert obs.shape == (3, ROWS, COLS)
        if done:
            break

    print()
    print(f"Final iron mask: {int(env.iron_mask.sum())} cells deposited")
    print("Smoke test PASSED.")
    if not _FEMM_AVAILABLE:
        print("(Reward/force values are zeros because pyfemm is not installed; "
              "flux channel is random noise.  Run on Windows with FEMM for real "
              "physics.)")


if __name__ == '__main__':
    _smoke_test()
