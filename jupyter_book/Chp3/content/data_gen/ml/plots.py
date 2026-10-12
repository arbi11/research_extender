"""Diagnostic plots for Phase 5."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def loss_curves(histories: Sequence[Dict[str, list]], out_png: Path, title: str = "") -> None:
    """Overlay train/val loss curves across folds."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(histories), 1)))
    for k, hist in enumerate(histories):
        epochs = np.arange(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], color=colors[k], linewidth=1.2,
                label=f"fold {k+1} train" if k < 10 else None)
        ax.plot(epochs, hist["val_loss"], color=colors[k], linewidth=1.2,
                linestyle="--",
                label=f"fold {k+1} val" if k < 10 else None)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (standardised targets)")
    ax.set_yscale("log")
    ax.set_title(title or "Phase 5A -- envelope-net training curves")
    ax.grid(True, alpha=0.4)
    if len(histories) <= 10:
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def map_comparison(
    oof_pred: np.ndarray,         # (N, H, W)
    oof_true: np.ndarray,         # (N, H, W), may contain NaN
    mask: np.ndarray,             # (N, H, W) bool
    out_png: Path,
    n_designs_to_show: int = 4,
    seed: int = 0,
) -> None:
    """Grid of (true, predicted, residual) triplets for n representative designs.

    Selects designs spanning the range of envelope size (sum of mask).
    Uses the same colour scheme as the Phase-4 efficiency map (jet, 40-100%).
    """
    n_total = oof_pred.shape[0]
    n_show = min(n_designs_to_show, n_total)
    if n_show == 0:
        return

    # Pick representative designs by spanning the masked-cell count (proxy for
    # envelope size, which correlates with design family).
    mask_counts = mask.reshape(n_total, -1).sum(axis=1)
    order = np.argsort(mask_counts)
    # Evenly sample n_show indices across the sorted range.
    picks = order[np.linspace(0, n_total - 1, n_show, dtype=int)]

    fig, axes = plt.subplots(n_show, 3, figsize=(13, 3.0 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(picks):
        m = mask[idx].astype(bool)
        true = np.where(m, oof_true[idx], np.nan) * 100.0       # show in %
        pred = np.where(m, oof_pred[idx], np.nan) * 100.0
        resid = np.where(m, (oof_pred[idx] - oof_true[idx]) * 100.0, np.nan)

        # Shared colour range for true & predicted to make panel comparison fair.
        finite_eta = np.concatenate([
            true[m & np.isfinite(true)].ravel(),
            pred[m & np.isfinite(pred)].ravel(),
        ]) if m.any() else np.array([90.0, 100.0])
        if finite_eta.size:
            vmin, vmax = float(np.nanpercentile(finite_eta, 2)), float(np.nanpercentile(finite_eta, 99.5))
        else:
            vmin, vmax = 40.0, 100.0
        rmax = float(np.nanmax(np.abs(resid))) if np.any(m) else 5.0
        rmax = max(rmax, 0.5)

        im0 = axes[row, 0].imshow(true, origin="lower", aspect="auto", cmap="jet",
                                  vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"design {idx}: true $\\eta$ [%]")
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.04)

        im1 = axes[row, 1].imshow(pred, origin="lower", aspect="auto", cmap="jet",
                                  vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"design {idx}: predicted $\\eta$ [%]")
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.04)

        im2 = axes[row, 2].imshow(resid, origin="lower", aspect="auto", cmap="RdBu_r",
                                  vmin=-rmax, vmax=+rmax)
        axes[row, 2].set_title(f"design {idx}: residual (pred - true) [%]")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.04)

        for ax in axes[row, :]:
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Phase 5B -- out-of-fold map predictions")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def pred_vs_actual(
    oof_pred: np.ndarray,
    oof_true: np.ndarray,
    target_names: Sequence[str],
    out_png: Path,
) -> None:
    """Per-target scatter of out-of-fold prediction vs ground truth."""
    n_targets = oof_pred.shape[1]
    fig, axes = plt.subplots(1, n_targets, figsize=(5.5 * n_targets, 5))
    if n_targets == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        true_k = oof_true[:, k]
        pred_k = oof_pred[:, k]
        lo = float(min(true_k.min(), pred_k.min()))
        hi = float(max(true_k.max(), pred_k.max()))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.6, linewidth=1)
        ax.scatter(true_k, pred_k, s=40, alpha=0.7, edgecolor="k", linewidth=0.4)
        ax.set_xlabel(f"True {target_names[k]}")
        ax.set_ylabel(f"Predicted {target_names[k]}")
        ax.set_title(target_names[k])
        ax.grid(True, alpha=0.4)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Phase 5A -- out-of-fold predictions vs ground truth")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
