"""Single source of truth for motor geometry, materials, and sweep grid.

Units: millimetres for geometry; SI for everything else.
Currents are PEAK amperes throughout the pipeline; conversions to RMS happen
only at plot time (RMS = peak / sqrt(2)).
"""

from dataclasses import dataclass, field
from typing import Literal, Tuple, Union
import numpy as np


@dataclass
class IPMGeometry:
    # Stator
    R_stator_outer: float = 100.0      # mm
    R_stator_bore: float = 60.0        # mm
    num_slots: int = 12
    slot_depth: float = 25.0           # mm (radial)
    slot_angle_deg: float = 8.0        # tangential opening
    turns_per_slot: int = 50

    # Rotor.
    # num_poles=4 matches the 12-slot ABC-ABC distributed winding pattern in
    # WindingPattern below: each phase's 4 coils land at electrical angles
    # 0/180/0/180 deg with alternating polarity, so all four EMFs add
    # constructively. The previous 10-pole choice (from the original notebook)
    # gave zero net PM flux linkage in every phase -- a pole/winding mismatch,
    # not a geometry issue. Plan FSCW 12/10 (Option B) as a separate machine
    # once the 12/4 IPM is validated end-to-end.
    R_rotor_outer: float = 59.0        # mm  (=> 1.0 mm air gap)
    R_rotor_inner: float = 30.0        # mm
    num_poles: int = 4

    # Magnets (interior, rectangular, alternating polarity).
    # magnet_length is RADIAL; with R_rotor_outer=59 and R_magnet_center=44.5
    # a 26 mm magnet leaves only 1.5 mm of back-iron on each side, which
    # saturates and forces PM flux across the air gap instead of leaking
    # tangentially through the rotor steel.
    magnet_length: float = 26.0        # mm (radial)
    magnet_width: float = 8.0          # mm (tangential)
    # Tangential air-slot flux barriers at each magnet end. Width = tangential
    # extent of each barrier (0 disables them).
    flux_barrier_width: float = 2.0    # mm (tangential, each end)

    # Mechanical-degree offset applied to every magnet position so an
    # open-circuit FEA solve at theta_elec_deg=0 yields lambda_d = +lambda_PM,
    # lambda_q = 0. For our 12-slot/4-pole full-pitch winding the analytical
    # result lambda_a(phi_PM) = -4*sin(2*phi_PM) is maximum POSITIVE at
    # phi_PM = 135 deg mech. (Naive guess of 45 deg gives lambda_d = -lambda_PM
    # -- same magnitude, wrong sign, which also flips the d-q torque sign.)
    rotor_offset_deg: float = 135.0

    # Stack
    stack_length_mm: float = 30.0      # axial depth; FEMM uses this directly

    @property
    def num_pole_pairs(self) -> int:
        return self.num_poles // 2

    @property
    def air_gap_mm(self) -> float:
        return self.R_stator_bore - self.R_rotor_outer


@dataclass
class FSCWGeometry:
    """12-slot/10-pole fractional-slot concentrated-winding PM motor with
    surface-mount magnets.

    Rotor has no interior cavities/barriers - magnets sit on the rotor surface.
    Stator winding is concentrated (one coil per tooth) with the EMETOR-standard
    AA'CC'BB'AA'CC'BB' double-layer-equivalent pattern; winding factor ~0.933.
    """
    # Stator (same shape as IPM)
    R_stator_outer: float = 100.0      # mm
    R_stator_bore: float = 60.0        # mm
    num_slots: int = 12
    slot_depth: float = 25.0           # mm (radial)
    slot_angle_deg: float = 8.0        # tangential opening
    turns_per_slot: int = 50

    # Rotor
    R_rotor_outer: float = 59.0        # mm  (=> 1.0 mm air gap)
    R_rotor_inner: float = 30.0        # mm
    num_poles: int = 10

    # Surface magnets (rectangular approximation of arc magnets).
    # `magnet_thickness` is radial extent (sits on the rotor outer surface);
    # `magnet_width` is tangential extent.
    magnet_thickness: float = 4.0      # mm (radial)
    magnet_width: float = 14.0         # mm (tangential)

    # Pole alignment: rotor offset so phase-A flux linkage is maximal positive
    # at the open-circuit FEA snapshot (electrical theta = 0). The exact value
    # depends on the winding pattern; we tune to 18 deg mech (-> 90 deg elec
    # for 5-pole-pair machine, putting first PM north pole on the d-axis).
    rotor_offset_deg: float = 18.0

    # Stack
    stack_length_mm: float = 30.0      # axial depth

    @property
    def num_pole_pairs(self) -> int:
        return self.num_poles // 2

    @property
    def air_gap_mm(self) -> float:
        return self.R_stator_bore - self.R_rotor_outer

    # For interface parity with IPMGeometry. The FSCW build doesn't draw
    # flux barriers but downstream code checks the attribute.
    @property
    def flux_barrier_width(self) -> float:
        return 0.0

    # Surface magnets have no "magnet_length" in the IPM sense; expose the
    # radial dimension under both names so design-space code can compose
    # uniformly.
    @property
    def magnet_length(self) -> float:
        return self.magnet_thickness


@dataclass
class Materials:
    stator_steel: str = "M-19 Steel"   # FEMM library
    rotor_steel: str = "M-19 Steel"
    winding: str = "Copper"
    magnet_name: str = "NdFeB_42"
    magnet_mu_r: float = 1.05
    magnet_Hc_Am: float = 890_000.0    # A/m  (NdFeB-42 typical, per pyFEMM manual)


@dataclass
class WindingPattern:
    # 12-slot, 4-pole IPM distributed winding (A C B A C B ...) with alternating polarity.
    # Distributed because IPM's 4-pole field needs phases spread across ~3 slots per pole per phase.
    phases: Tuple[str, ...] = ("A", "C", "B", "A", "C", "B", "A", "C", "B", "A", "C", "B")
    polarities: Tuple[int, ...] = (+1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -1)


@dataclass
class FSCWWindingPattern:
    """12-slot/10-pole FSCW double-layer-equivalent pattern. Winding factor ~0.933.

    Phase layout (which slot belongs to which phase) is the EMETOR
    AA'CC'BB'AA'CC'BB' atlas pattern. The crucial *polarity* sequence has
    the second pair of coils for each phase wound in the OPPOSITE direction
    from the first pair (slots 6-11 polarities are the negation of slots 0-5).

    Why the second-half polarity flip is non-negotiable. The two A-coils sit
    around tooth 1 (slots 0-1, axis at 15 deg mech) and tooth 7 (slots 6-7,
    axis at 195 deg mech) - exactly 180 deg apart mechanically. With p=5
    pole pairs that becomes 180 deg ELECTRICAL at the rotor's working
    harmonic, so two same-direction coils CANCEL at p=5. Reversing the
    second coil (flipping polarities of slots 6-11) makes them add at p=5,
    giving the required 10-pole MMF distribution. Without this flip, phase A
    drives only even-pole-pair MMFs and produces no working flux against a
    10-pole rotor (a previous version of this code shipped the
    non-flipped form and produced ~1 Nm of stray torque at 50 A peak
    instead of the expected several Nm).
    """
    phases: Tuple[str, ...] = ("A", "A", "C", "C", "B", "B", "A", "A", "C", "C", "B", "B")
    polarities: Tuple[int, ...] = (+1, -1, +1, -1, +1, -1, -1, +1, -1, +1, -1, +1)


@dataclass
class SweepGrid:
    """Sparse 3x3 grid in (i_d, i_q) for Stage-1 characterization.

    The paper builds the lambda surfaces from 9 FEA samples + 2nd-degree poly,
    not from a dense brute-force grid.
    """
    i_d_peak_A: Tuple[float, ...] = (-100.0, -50.0, 0.0)   # IPM: i_d <= 0 typical
    i_q_peak_A: Tuple[float, ...] = (0.0, 50.0, 100.0)
    theta_elec_deg: float = 0.0                            # d-axis aligned with phase A


@dataclass
class FEMMSolver:
    problem_type: int = 0              # 0 = magnetostatic
    units: str = "millimeters"
    symmetry: str = "planar"
    precision: float = 1e-8
    min_angle_deg: float = 30
    air_gap_mesh_mm: float = 0.5       # finer mesh in the gap
    boundary_size_factor: float = 3.0  # air box = factor * stator OD
    # Hide the FEMM main window. True is preferred for headless / parallel
    # sweeps; flip to False if you want to inspect the geometry visually.
    hide_window: bool = True


@dataclass
class MachineLimits:
    """Drive-side limits used by the Phase 2 control-strategy solver.

    Currents are PEAK amperes (matching the sweep grid).
    V_max_peak_V is peak per-phase voltage; with sinusoidal PWM and a 350 V
    DC bus, V_DC * 1/sqrt(3) ~ 200 V is a typical envelope.
    """
    I_max_peak_A: float = 100.0
    V_max_peak_V: float = 200.0
    R_s_ohm: float = 0.05              # per-phase resistance (rough)
    max_speed_rpm: float = 8000.0


@dataclass
class Config:
    geom: Union[IPMGeometry, FSCWGeometry] = field(default_factory=IPMGeometry)
    mat: Materials = field(default_factory=Materials)
    winding: Union[WindingPattern, FSCWWindingPattern] = field(default_factory=WindingPattern)
    sweep: SweepGrid = field(default_factory=SweepGrid)
    solver: FEMMSolver = field(default_factory=FEMMSolver)
    limits: MachineLimits = field(default_factory=MachineLimits)
    motor_type: Literal["ipm", "fscw"] = "ipm"


def i_d_q_grid(sweep: SweepGrid) -> np.ndarray:
    """Return shape (N, 2) array of (i_d, i_q) peak-amp pairs."""
    pts = [(d, q) for d in sweep.i_d_peak_A for q in sweep.i_q_peak_A]
    return np.array(pts, dtype=float)
