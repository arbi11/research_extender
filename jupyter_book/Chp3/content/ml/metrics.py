"""Regression metrics (numpy)."""

from __future__ import annotations
import numpy as np


def _to_np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def rmse(pred, true, axis=None) -> np.ndarray | float:
    """Root-mean-square error. Returns scalar if axis is None else per-axis vector."""
    p, t = _to_np(pred), _to_np(true)
    return float(np.sqrt(np.mean((p - t) ** 2, axis=axis))) if axis is None else \
        np.sqrt(np.mean((p - t) ** 2, axis=axis))


def mae(pred, true, axis=None):
    p, t = _to_np(pred), _to_np(true)
    return float(np.mean(np.abs(p - t), axis=axis)) if axis is None else \
        np.mean(np.abs(p - t), axis=axis)


def mape(pred, true, eps: float = 1e-6, axis=None):
    """Mean absolute percentage error (in %). Guards against zero divisor."""
    p, t = _to_np(pred), _to_np(true)
    denom = np.where(np.abs(t) < eps, eps, np.abs(t))
    val = np.mean(np.abs(p - t) / denom, axis=axis) * 100.0
    return float(val) if axis is None else val


def r2(pred, true, axis=None):
    """Coefficient of determination."""
    p, t = _to_np(pred), _to_np(true)
    ss_res = np.sum((t - p) ** 2, axis=axis)
    ss_tot = np.sum((t - np.mean(t, axis=axis, keepdims=True)) ** 2, axis=axis)
    val = 1.0 - ss_res / np.where(ss_tot < 1e-12, 1.0, ss_tot)
    return float(val) if axis is None else val


# -----------------------------------------------------------------------------
# Masked variants (Phase 5B: NaN cells outside the operating envelope)
# -----------------------------------------------------------------------------

def masked_rmse(pred, true, mask):
    p, t = _to_np(pred), _to_np(true)
    m = _to_np(mask).astype(bool)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean(((p - t) ** 2)[m])))


def masked_mae(pred, true, mask):
    p, t = _to_np(pred), _to_np(true)
    m = _to_np(mask).astype(bool)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(p - t)[m]))


def masked_r2(pred, true, mask):
    p, t = _to_np(pred), _to_np(true)
    m = _to_np(mask).astype(bool)
    if not m.any():
        return float("nan")
    pm = p[m]
    tm = t[m]
    ss_res = float(np.sum((tm - pm) ** 2))
    ss_tot = float(np.sum((tm - tm.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)
