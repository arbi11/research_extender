"""Park / inverse-Park transforms (amplitude-invariant convention).

The amplitude-invariant inverse Park used here gives:
    peak phase current magnitude = sqrt(i_d^2 + i_q^2)

i_a, i_b, i_c always sum to zero (no neutral current).
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def inverse_park(i_d: float, i_q: float, theta_elec_deg: float = 0.0) -> Tuple[float, float, float]:
    """d-q peak currents -> a, b, c phase currents (peak).

    Sign on i_q is chosen so the standard motoring formula
        T_em = (3/2) * p * (lambda_d * i_q - lambda_q * i_d)
    matches the FEMM Maxwell-stress torque sign for our winding/rotor convention.
    The original "-i_q*sin(theta)" form (Krause-style) gave the opposite sign
    for our positive-sequence ABC layout, so the q-axis direction is flipped here.
    """
    theta = np.radians(theta_elec_deg)
    i_a = i_d * np.cos(theta) + i_q * np.sin(theta)
    i_b = i_d * np.cos(theta - 2 * np.pi / 3) + i_q * np.sin(theta - 2 * np.pi / 3)
    i_c = i_d * np.cos(theta + 2 * np.pi / 3) + i_q * np.sin(theta + 2 * np.pi / 3)
    return float(i_a), float(i_b), float(i_c)


def park_flux(lambda_a: float, lambda_b: float, lambda_c: float,
              theta_elec_deg: float = 0.0) -> Tuple[float, float]:
    """3-phase flux linkages -> (lambda_d, lambda_q) in d-q frame.

    Clarke: lambda_alpha = lambda_a;  lambda_beta = (lambda_a + 2*lambda_b) / sqrt(3)
    Park:   lambda_d =  lambda_alpha cos(theta) + lambda_beta sin(theta)
            lambda_q = -lambda_alpha sin(theta) + lambda_beta cos(theta)
    """
    theta = np.radians(theta_elec_deg)
    lambda_alpha = lambda_a
    lambda_beta = (lambda_a + 2.0 * lambda_b) / np.sqrt(3.0)
    lambda_d = lambda_alpha * np.cos(theta) + lambda_beta * np.sin(theta)
    # q-axis sign matches the flipped convention in inverse_park.
    lambda_q = lambda_alpha * np.sin(theta) - lambda_beta * np.cos(theta)
    return float(lambda_d), float(lambda_q)


def validate_sum_zero(i_a: float, i_b: float, i_c: float, tol: float = 1e-6) -> None:
    s = abs(i_a + i_b + i_c)
    if s > tol:
        raise ValueError(f"Phase currents do not sum to zero: {s:.3e}")
