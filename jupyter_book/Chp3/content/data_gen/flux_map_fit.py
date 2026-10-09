"""2nd-degree bivariate polynomial fits for lambda_d(i_d, i_q) and lambda_q(i_d, i_q).

Matches the paper's Stage-1 post-processing: a small number of FEA samples are
fitted by a 2nd-degree polynomial; the surface is then evaluated on a dense
grid for visualization and use in downstream MTPA / FW / MTPV control solvers.

Polynomial form (6 coefficients):
    f(d, q) = c00 + c10*d + c01*q + c20*d^2 + c11*d*q + c02*q^2
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def _design_matrix(d: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Columns: 1, d, q, d^2, d*q, q^2."""
    return np.column_stack([
        np.ones_like(d), d, q, d * d, d * q, q * q,
    ])


@dataclass
class PolyFit2D:
    coeffs: np.ndarray  # shape (6,) — [c00, c10, c01, c20, c11, c02]

    def __call__(self, d: np.ndarray, q: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        q = np.asarray(q, dtype=float)
        X = _design_matrix(d.ravel(), q.ravel())
        return (X @ self.coeffs).reshape(d.shape)


def fit_poly2(i_d: np.ndarray, i_q: np.ndarray, values: np.ndarray) -> PolyFit2D:
    """Least-squares 2D 2nd-degree polynomial fit."""
    i_d = np.asarray(i_d, dtype=float).ravel()
    i_q = np.asarray(i_q, dtype=float).ravel()
    y = np.asarray(values, dtype=float).ravel()
    if not (len(i_d) == len(i_q) == len(y)):
        raise ValueError("i_d, i_q, values must have the same length")
    if len(y) < 6:
        raise ValueError(f"Need at least 6 samples for 2nd-degree fit, got {len(y)}")
    X = _design_matrix(i_d, i_q)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return PolyFit2D(coeffs=coeffs)


def fit_flux_maps(
    df,  # pd.DataFrame from stage1_characterize.run_sparse_sweep
) -> Tuple[PolyFit2D, PolyFit2D]:
    """Return (lambda_d_fit, lambda_q_fit) from a sample DataFrame."""
    lam_d_fit = fit_poly2(df["i_d_A"].values, df["i_q_A"].values, df["lambda_d_Wb"].values)
    lam_q_fit = fit_poly2(df["i_d_A"].values, df["i_q_A"].values, df["lambda_q_Wb"].values)
    return lam_d_fit, lam_q_fit


def fit_torque(df) -> PolyFit2D:
    """2D-poly fit of the FEMM Maxwell-stress torque T(i_d, i_q).

    Used INSTEAD of the d-q torque formula T=(3/2)*p*(lambda_d*i_q-lambda_q*i_d).
    The latter is a fundamental-only approximation and undercounts torque a lot
    for machines with non-sinusoidal flux (e.g. rectangular interior magnets).
    """
    return fit_poly2(df["i_d_A"].values, df["i_q_A"].values, df["torque_Nm"].values)


def fit_quality(fit: PolyFit2D, i_d: np.ndarray, i_q: np.ndarray, y: np.ndarray) -> dict:
    """Basic goodness-of-fit metrics on the training samples (sanity check)."""
    y_hat = fit(np.asarray(i_d), np.asarray(i_q))
    resid = np.asarray(y) - y_hat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((np.asarray(y) - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "max_abs": float(np.max(np.abs(resid))),
        "r2": r2,
    }
