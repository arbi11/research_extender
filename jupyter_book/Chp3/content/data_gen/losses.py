"""Loss models used in Stage 3.

Copper loss: analytical from i_d, i_q and per-phase resistance.

Iron (core) loss: simplified Steinmetz model driven by FEMM-computed B-field
amplitudes in the stator iron. This is a lumped/approximate version; a proper
calculation would step the rotor through one electrical period and integrate
the Steinmetz density at every iron element. For v1 we use:

    P_Fe = (k_h * f * <B^a> + k_e * f^2 * <B^2>) * V_iron

where <.> denotes spatial average over the stator iron volume at one rotor
snapshot, and f = p * N_rpm / 60 is the electrical frequency.

Constants for typical M-19 silicon steel (literature values, room temperature):
    k_h ~ 0.02   [W/(kg * Hz * T^a)]
    k_e ~ 0.0001 [W/(kg * Hz^2 * T^2)]
    a   ~ 1.8 .. 2.0
Densities convert per-kg to per-m^3 via rho ~ 7650 kg/m^3.

For our small 12-slot/4-pole machine (stack 30 mm), rated iron loss should sit
in the few-watts to tens-of-watts range at thousands of rpm.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class IronLossParams:
    k_h_W_per_kg_Hz_Ta: float = 0.02
    k_e_W_per_kg_Hz2_T2: float = 0.0001
    steinmetz_a: float = 1.8
    rho_steel_kg_per_m3: float = 7650.0


def copper_loss_W(i_d: float, i_q: float, R_s_ohm: float) -> float:
    """3-phase copper loss for amplitude-invariant d-q currents (PEAK values).

    P_Cu_3ph = 3 * R_s * I_rms_per_phase^2
    With I_rms = I_peak / sqrt(2) and I_peak = sqrt(i_d^2 + i_q^2):
    P_Cu_3ph = (3/2) * R_s * (i_d^2 + i_q^2)
    """
    return 1.5 * R_s_ohm * (i_d * i_d + i_q * i_q)


def iron_loss_W(
    B_peak_T: float,
    f_elec_Hz: float,
    V_iron_m3: float,
    params: IronLossParams = IronLossParams(),
) -> float:
    """Lumped Steinmetz iron loss given a representative B_peak (T)."""
    if f_elec_Hz <= 0 or V_iron_m3 <= 0 or B_peak_T <= 0:
        return 0.0
    mass_kg = params.rho_steel_kg_per_m3 * V_iron_m3
    hyst = params.k_h_W_per_kg_Hz_Ta * f_elec_Hz * (B_peak_T ** params.steinmetz_a)
    eddy = params.k_e_W_per_kg_Hz2_T2 * (f_elec_Hz * B_peak_T) ** 2
    return float((hyst + eddy) * mass_kg)


def electrical_frequency_Hz(N_rpm: float, pole_pairs: int) -> float:
    return pole_pairs * N_rpm / 60.0


def efficiency(T_em_Nm: float, omega_m_rad_s: float, P_Cu_W: float, P_Fe_W: float) -> float:
    """Motoring efficiency = P_mech / (P_mech + P_Cu + P_Fe).

    Returns 0.0 if the operating point produces no shaft power.
    """
    P_mech = T_em_Nm * omega_m_rad_s
    if P_mech <= 0:
        return 0.0
    P_total = P_mech + P_Cu_W + P_Fe_W
    return float(P_mech / P_total)


def omega_mech_rad_s(N_rpm: float) -> float:
    return 2.0 * np.pi * N_rpm / 60.0


def stator_iron_volume_m3(geom) -> float:
    """Approximate stator back-iron + tooth volume (excluding slots).

    Stator back-iron is an annulus from R_slot_bottom to R_stator_outer.
    Teeth are approximated as filling (1 - slot_fill) of the annulus from
    R_stator_bore to R_slot_bottom.
    """
    R_so = geom.R_stator_outer * 1e-3
    R_sb = geom.R_stator_bore * 1e-3
    R_slot_bot = (geom.R_stator_bore + geom.slot_depth) * 1e-3
    L = geom.stack_length_mm * 1e-3
    # Back-iron annulus
    A_back = np.pi * (R_so ** 2 - R_slot_bot ** 2)
    # Tooth region: annulus minus slot volume (approximating slots as wedges)
    slot_angular_fraction = (geom.num_slots * np.radians(geom.slot_angle_deg)) / (2 * np.pi)
    A_tooth_annulus = np.pi * (R_slot_bot ** 2 - R_sb ** 2) * (1.0 - slot_angular_fraction)
    return (A_back + A_tooth_annulus) * L
