"""Shared training loop."""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from torch import nn


log = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)


def train_one_fold(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    log_every: int = 50,
    fold_name: str = "fold",
) -> TrainHistory:
    """Full-batch Adam + MSE training. Mutates `model` in place.

    Tiny dataset (~40 samples), so we forgo mini-batching: each epoch is one
    forward + backward over the full training set.
    """
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    train_x_t = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    train_y_t = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    val_x_t = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    val_y_t = torch.as_tensor(val_y, dtype=torch.float32, device=device)

    hist = TrainHistory()
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(train_x_t)
        loss = mse(pred, train_y_t)
        loss.backward()
        opt.step()
        hist.train_loss.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(mse(model(val_x_t), val_y_t).item())
        hist.val_loss.append(val_loss)

        if epoch == 1 or epoch == epochs or (log_every and epoch % log_every == 0):
            log.info("  [%s] ep %4d/%d  train=%.5f  val=%.5f",
                     fold_name, epoch, epochs, hist.train_loss[-1], val_loss)
    return hist


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """Forward pass; returns numpy on CPU."""
    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x, dtype=torch.float32)
        return model(x_t).cpu().numpy()


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE evaluated only where mask is True (1). NaN-safe."""
    sq = (pred - target) ** 2 * mask
    return sq.sum() / mask.sum().clamp_min(1.0)


def train_one_fold_masked(
    model: nn.Module,
    train_x: np.ndarray,        # (n_train, n_params)
    train_y: np.ndarray,        # (n_train, H, W) NaN-filled outside envelope
    train_mask: np.ndarray,     # (n_train, H, W) bool
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_mask: np.ndarray,
    *,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    log_every: int = 50,
    fold_name: str = "fold",
) -> TrainHistory:
    """Training loop for the map net.

    Targets may contain NaN outside the envelope; we replace them with 0 in the
    tensor and rely on the mask to zero those positions in the loss. Full-batch
    Adam + masked MSE.
    """
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_y_clean = np.where(train_mask, train_y, 0.0)
    val_y_clean = np.where(val_mask, val_y, 0.0)

    train_x_t = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    train_y_t = torch.as_tensor(train_y_clean, dtype=torch.float32, device=device)
    train_m_t = torch.as_tensor(train_mask, dtype=torch.float32, device=device)
    val_x_t = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    val_y_t = torch.as_tensor(val_y_clean, dtype=torch.float32, device=device)
    val_m_t = torch.as_tensor(val_mask, dtype=torch.float32, device=device)

    hist = TrainHistory()
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(train_x_t)
        loss = _masked_mse(pred, train_y_t, train_m_t)
        loss.backward()
        opt.step()
        hist.train_loss.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x_t)
            val_loss = float(_masked_mse(val_pred, val_y_t, val_m_t).item())
        hist.val_loss.append(val_loss)

        if epoch == 1 or epoch == epochs or (log_every and epoch % log_every == 0):
            log.info("  [%s] ep %4d/%d  train=%.5f  val=%.5f",
                     fold_name, epoch, epochs, hist.train_loss[-1], val_loss)
    return hist
