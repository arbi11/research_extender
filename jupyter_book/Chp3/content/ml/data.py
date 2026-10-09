"""Master-dataset loader, CV splitter, and standardisation helpers.

No torch import here — keeps `_smoke_test.py` independent of the DL framework.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class DatasetBundle:
    design_params: np.ndarray       # (N, 6) float
    N_max_rpm: np.ndarray           # (N,)   float
    T_max_Nm: np.ndarray            # (N,)   float
    eta_norm_grids: np.ndarray      # (N, 60, 80) float, NaN outside envelope
    eta_mask: np.ndarray            # (N, 60, 80) bool, True inside envelope
    param_names: List[str]
    N_norm_grid: np.ndarray         # (80,)
    T_norm_grid: np.ndarray         # (60,)

    @property
    def n_designs(self) -> int:
        return self.design_params.shape[0]


def load_dataset(npz_path: Path | str) -> DatasetBundle:
    """Load the master dataset NPZ written by data_gen.dataset_runner.aggregate()."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Master dataset not found at {npz_path}")
    z = np.load(npz_path, allow_pickle=False)

    eta = np.asarray(z["eta_norm_grids"], dtype=np.float32)
    mask = ~np.isnan(eta)
    return DatasetBundle(
        design_params=np.asarray(z["design_params"], dtype=np.float32),
        N_max_rpm=np.asarray(z["N_max_rpm"], dtype=np.float32),
        T_max_Nm=np.asarray(z["T_max_Nm"], dtype=np.float32),
        eta_norm_grids=eta,
        eta_mask=mask,
        param_names=[str(s) for s in z["param_names"]],
        N_norm_grid=np.asarray(z["N_norm_grid"], dtype=np.float32),
        T_norm_grid=np.asarray(z["T_norm_grid"], dtype=np.float32),
    )


# -----------------------------------------------------------------------------
# Standardisation
# -----------------------------------------------------------------------------

def standardise(
    x: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_std, mean, std). If mean/std given, reuse them (val/test path)."""
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
        # Avoid divide-by-zero for any constant column.
        std = np.where(std < 1e-12, 1.0, std)
    return (x - mean) / std, mean, std


def unstandardise(x_std: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x_std * std + mean


# -----------------------------------------------------------------------------
# Cross-validation splits
# -----------------------------------------------------------------------------

def kfold_indices(n: int, k: int, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """k-fold CV. Returns a list of (train_idx, val_idx) pairs.

    Shuffles indices once with `seed`, then partitions into k contiguous folds.
    Each fold's val set has size approximately n/k.
    """
    if k <= 1 or k > n:
        raise ValueError(f"kfold k must satisfy 2 <= k <= n; got k={k}, n={n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for sz in fold_sizes:
        val = perm[start : start + sz]
        train = np.concatenate([perm[:start], perm[start + sz :]])
        folds.append((np.sort(train), np.sort(val)))
        start += sz
    return folds


def loo_indices(n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Leave-one-out CV. Returns n (train_idx, val_idx) pairs."""
    return [(np.array([j for j in range(n) if j != i]), np.array([i])) for i in range(n)]
