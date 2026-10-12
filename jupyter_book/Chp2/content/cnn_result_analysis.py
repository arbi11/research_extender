"""
Error analysis helpers for CNN field predictions (used by 04c_cnn_training.ipynb).

Reconstructed: the original cnn_result_analysis.py is not in the Notion export.
The function names, arguments and returned keys follow how 04c calls them.

Fields are arrays of |B| (any consistent unit; 04c passes |B| / B_max).
Material masks use the labels from femm_solver.MeshInterpolator.generate_semantic_mask:
0 = air, 1 = iron, 2 = copper, 3 = magnet.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

MATERIAL_LABELS = {0: 'Air', 1: 'Iron', 2: 'Copper', 3: 'Magnet'}


def compute_comprehensive_metrics(pred_fields, gt_fields):
    """Global error metrics over all samples and pixels.

    Returns a dict with nmse_percent, rmse, mae, percentile_95, max_error,
    r2 and relative_l2_percent.
    """
    pred = np.asarray(pred_fields, dtype=np.float64)
    gt = np.asarray(gt_fields, dtype=np.float64)
    err = pred - gt
    abs_err = np.abs(err)

    mse = float(np.mean(err ** 2))
    gt_power = float(np.mean(gt ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2))

    return {
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'mae': float(np.mean(abs_err)),
        'nmse_percent': 100.0 * mse / gt_power if gt_power > 0 else float('nan'),
        'percentile_95': float(np.percentile(abs_err, 95)),
        'max_error': float(abs_err.max()),
        'r2': 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float('nan'),
        'relative_l2_percent': 100.0 * float(np.linalg.norm(err) / np.linalg.norm(gt))
        if np.linalg.norm(gt) > 0 else float('nan'),
    }


def compute_spatial_error_map(pred_field, gt_field, eps=1e-8):
    """Per-pixel error maps for one sample.

    Returns (signed_error, absolute_error, relative_error), where the
    relative error is |error| divided by the sample's peak |gt|.
    """
    pred = np.asarray(pred_field, dtype=np.float64)
    gt = np.asarray(gt_field, dtype=np.float64)
    err = pred - gt
    abs_err = np.abs(err)
    rel_err = abs_err / (np.abs(gt).max() + eps)
    return err, abs_err, rel_err


def identify_high_error_regions(abs_error_maps, material_masks, threshold_percentile=95):
    """Break absolute error down by material region.

    A pixel is "high error" if it exceeds the given percentile of all
    absolute errors. For each material present, returns mean_error,
    max_error, median_error, pixel_fraction (share of all pixels) and
    high_error_fraction (share of that material's pixels above threshold).
    """
    abs_err = np.asarray(abs_error_maps, dtype=np.float64)
    masks = np.rint(np.asarray(material_masks)).astype(int)
    threshold = np.percentile(abs_err, threshold_percentile)

    breakdown = {}
    for label, name in MATERIAL_LABELS.items():
        sel = masks == label
        if not sel.any():
            continue
        vals = abs_err[sel]
        breakdown[name] = {
            'mean_error': float(vals.mean()),
            'max_error': float(vals.max()),
            'median_error': float(np.median(vals)),
            'pixel_fraction': float(sel.mean()),
            'high_error_fraction': float(np.mean(vals > threshold)),
        }
    return breakdown


def _save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    return path


def plot_error_distribution(signed_error_maps, problem_type, out_dir):
    """Histogram of signed pixel errors (linear and log count)."""
    err = np.asarray(signed_error_maps).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, log in zip(axes, (False, True)):
        ax.hist(err, bins=200, color='steelblue', alpha=0.85, log=log)
        ax.axvline(0, color='k', lw=1)
        ax.axvline(err.mean(), color='r', ls='--', lw=1.2, label=f'mean = {err.mean():.2e}')
        ax.set_xlabel('Prediction − ground truth')
        ax.set_ylabel('Pixel count' + (' (log)' if log else ''))
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_title(f'{problem_type}: signed error distribution')
    axes[1].set_title(f'std = {err.std():.2e}')
    plt.tight_layout()
    return _save(fig, out_dir, f'{problem_type}_error_distribution.png')


def plot_average_error_map(abs_error_maps, problem_type, out_dir):
    """Mean and 95th-percentile absolute error at each pixel across samples."""
    abs_err = np.asarray(abs_error_maps)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in zip(
        axes,
        (abs_err.mean(axis=0), np.percentile(abs_err, 95, axis=0)),
        ('Mean |error|', '95th percentile |error|'),
    ):
        im = ax.imshow(data, origin='lower', cmap='hot')
        ax.set_title(f'{problem_type}: {title}')
        ax.set_xlabel('x-pixel')
        ax.set_ylabel('y-pixel')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return _save(fig, out_dir, f'{problem_type}_average_error_map.png')


def plot_error_vs_magnitude(pred_fields, gt_fields, out_dir, n_bins=30, max_points=50000, seed=0):
    """Absolute error against true field magnitude, plus predicted vs true."""
    pred = np.asarray(pred_fields).ravel()
    gt = np.asarray(gt_fields).ravel()
    abs_err = np.abs(pred - gt)

    rng = np.random.default_rng(seed)
    idx = rng.choice(gt.size, size=min(max_points, gt.size), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(gt[idx], abs_err[idx], s=2, alpha=0.2, color='steelblue')
    edges = np.linspace(gt.min(), gt.max(), n_bins + 1)
    which = np.clip(np.digitize(gt, edges) - 1, 0, n_bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.array([abs_err[which == b].mean() if np.any(which == b) else np.nan
                      for b in range(n_bins)])
    ax.plot(centers, means, 'r-', lw=2, label='binned mean')
    ax.set_xlabel('True |B|')
    ax.set_ylabel('|error|')
    ax.set_title('Error vs field magnitude')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.scatter(gt[idx], pred[idx], s=2, alpha=0.2, color='steelblue')
    lo, hi = float(min(gt.min(), pred.min())), float(max(gt.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_xlabel('True |B|')
    ax.set_ylabel('Predicted |B|')
    ax.set_title('Predicted vs true (pixels)')
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    return _save(fig, out_dir, 'error_vs_magnitude.png')
