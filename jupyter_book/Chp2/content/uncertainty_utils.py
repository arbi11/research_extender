"""
Utility functions for uncertainty quantification visualization.

This module contains plotting and analysis utilities used in Section 5
(Uncertainty Quantification) to keep the main notebook focused on concepts
rather than implementation details.

Author: Generated for Chapter 2, Section 5
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_calibration_analysis(mean_pred, std_pred, y_true,
                               confidence_levels=None,
                               model_name="MC Dropout"):
    """
    Create comprehensive calibration visualizations.

    Generates three plots:
    1. Reliability diagram (predicted vs observed confidence)
    2. Coverage comparison (expected vs observed)
    3. Sharpness analysis (confidence interval widths)

    Args:
        mean_pred: Predicted means (torch.Tensor, shape: [n_samples, 1])
        std_pred: Predicted uncertainties (torch.Tensor, shape: [n_samples, 1])
        y_true: Ground truth values (torch.Tensor, shape: [n_samples, 1])
        confidence_levels: List of confidence levels to evaluate (default: [0.50, 0.68, 0.90, 0.95, 0.99])
        model_name: Model name for plot titles

    Returns:
        dict: Calibration metrics containing:
            - coverages: Dict mapping confidence levels to observed coverage
            - interval_widths: Dict mapping confidence levels to mean interval width
            - mean_calibration_error: Average absolute calibration error
            - calibration_quality: String assessment ("Excellent", "Good", or "Needs Improvement")
    """
    if confidence_levels is None:
        confidence_levels = [0.50, 0.68, 0.90, 0.95, 0.99]

    z_scores = {
        0.50: 0.674,  # 50% CI: ±0.674σ
        0.68: 1.000,  # 68% CI: ±1.000σ (1 standard deviation)
        0.90: 1.645,  # 90% CI: ±1.645σ
        0.95: 1.960,  # 95% CI: ±1.960σ
        0.99: 2.576   # 99% CI: ±2.576σ
    }

    coverages = {}
    interval_widths = {}

    for conf_level in confidence_levels:
        z = z_scores[conf_level]

        # Calculate confidence interval bounds
        lower = mean_pred - z * std_pred
        upper = mean_pred + z * std_pred

        # Check if true values fall within interval
        within_interval = ((y_true >= lower) & (y_true <= upper)).float()
        observed_coverage = within_interval.mean().item()

        coverages[conf_level] = observed_coverage
        interval_widths[conf_level] = (upper - lower).mean().item()

    # Calculate calibration error
    calibration_errors = [abs(coverages[c] - c) for c in confidence_levels]
    mean_calibration_error = np.mean(calibration_errors)

    # Create visualization
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, wspace=0.3)

    # Plot 1: Reliability Diagram
    ax1 = fig.add_subplot(gs[0, 0])

    conf_levels_pct = [c * 100 for c in confidence_levels]
    observed_coverage_pct = [coverages[c] * 100 for c in confidence_levels]

    ax1.plot([0, 100], [0, 100], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)
    ax1.plot(conf_levels_pct, observed_coverage_pct, 'bo-', linewidth=2, markersize=8,
            label=f'{model_name}')

    # Add error bars showing deviation from ideal
    for i, (pred, obs) in enumerate(zip(conf_levels_pct, observed_coverage_pct)):
        if abs(pred - obs) > 3:
            ax1.plot([pred, pred], [pred, obs], 'r-', linewidth=1.5, alpha=0.5)

    # Shade acceptable region (±5% from diagonal)
    ax1.fill_between([0, 100], [0, 100], [5, 105], alpha=0.1, color='green',
                    label='Acceptable (±5%)')
    ax1.fill_between([0, 100], [0, 100], [-5, 95], alpha=0.1, color='green')

    ax1.set_xlabel('Predicted Confidence Level (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Observed Coverage (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Reliability Diagram', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 100])

    # Add calibration quality annotation
    if mean_calibration_error < 0.05:
        quality = "Excellent"
        color = "green"
    elif mean_calibration_error < 0.10:
        quality = "Good"
        color = "blue"
    else:
        quality = "Needs Improvement"
        color = "red"

    ax1.text(0.05, 0.95, f'Calibration: {quality}\nMean Error: {mean_calibration_error*100:.1f}%',
            transform=ax1.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3),
            verticalalignment='top')

    # Plot 2: Coverage at Different Confidence Levels
    ax2 = fig.add_subplot(gs[0, 1])

    x = np.arange(len(confidence_levels))
    width = 0.35

    bars1 = ax2.bar(x - width/2, conf_levels_pct, width, label='Expected',
                   color='gray', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x + width/2, observed_coverage_pct, width, label='Observed',
                   color='steelblue', alpha=0.7, edgecolor='black')

    ax2.set_xlabel('Confidence Level', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Coverage (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Expected vs. Observed Coverage', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{int(c*100)}%' for c in confidence_levels])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (expected, observed) in enumerate(zip(conf_levels_pct, observed_coverage_pct)):
        diff = observed - expected
        color_text = 'green' if abs(diff) < 5 else 'red'
        ax2.text(i, max(expected, observed) + 2, f'{diff:+.1f}%',
                ha='center', fontsize=9, fontweight='bold', color=color_text)

    # Plot 3: Sharpness Analysis
    ax3 = fig.add_subplot(gs[0, 2])

    widths = [interval_widths[c] for c in confidence_levels]
    bars3 = ax3.bar(x, widths, color='coral', alpha=0.7, edgecolor='black')

    ax3.set_xlabel('Confidence Level', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Average Interval Width', fontsize=11, fontweight='bold')
    ax3.set_title('Sharpness: Confidence Interval Widths', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{int(c*100)}%' for c in confidence_levels])
    ax3.grid(True, alpha=0.3, axis='y')

    # Add width labels
    for i, width in enumerate(widths):
        ax3.text(i, width, f'{width:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    plt.suptitle(f'Calibration Analysis: {model_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return {
        'coverages': coverages,
        'interval_widths': interval_widths,
        'mean_calibration_error': mean_calibration_error,
        'calibration_quality': quality
    }


def plot_uncertainty_vs_error(std_pred, errors, title="Uncertainty vs Error", ax=None):
    """
    Plot scatter plot showing correlation between predicted uncertainty and actual error.

    Args:
        std_pred: Predicted uncertainties (numpy array or torch.Tensor)
        errors: Actual prediction errors (numpy array or torch.Tensor)
        title: Plot title
        ax: Matplotlib axis (creates new if None)

    Returns:
        float: Correlation coefficient between uncertainty and error
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Convert to numpy if needed
    if isinstance(std_pred, torch.Tensor):
        std_pred = std_pred.numpy().flatten()
    if isinstance(errors, torch.Tensor):
        errors = errors.numpy().flatten()

    # Scatter plot
    ax.scatter(std_pred, errors, alpha=0.6, s=30)
    ax.set_xlabel('Predicted Uncertainty (σ)', fontsize=10)
    ax.set_ylabel('Absolute Error |ŷ - y|', fontsize=10)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(std_pred, errors, 1)
    p = np.poly1d(z)
    ax.plot(std_pred, p(std_pred), "r--", alpha=0.8, linewidth=2)

    # Calculate and display correlation
    corr = np.corrcoef(std_pred, errors)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='top')

    return corr


def print_calibration_metrics(coverages, interval_widths, mean_calibration_error,
                               confidence_levels=None):
    """
    Print formatted calibration metrics table.

    Args:
        coverages: Dict mapping confidence levels to observed coverage
        interval_widths: Dict mapping confidence levels to mean interval width
        mean_calibration_error: Mean calibration error
        confidence_levels: List of confidence levels (default: keys from coverages)
    """
    if confidence_levels is None:
        confidence_levels = sorted(coverages.keys())

    print(f"\n  Calibration Metrics:")
    print(f"  " + "=" * 66)
    print(f"  {'Confidence Level':<20} {'Expected':>12} {'Observed':>12} {'Error':>12} {'Width':>10}")
    print(f"  " + "-" * 66)

    for conf_level in confidence_levels:
        expected = conf_level * 100
        observed = coverages[conf_level] * 100
        error = observed - expected
        width = interval_widths[conf_level]

        status = "✓" if abs(error) < 5 else "✗"
        print(f"  {status} {conf_level*100:>4.0f}%{'':<14} {expected:>11.1f}% {observed:>11.1f}% {error:>+11.1f}% {width:>10.3f}")

    print(f"  " + "=" * 66)
    print(f"  Mean Calibration Error: {mean_calibration_error*100:.2f}%")


def plot_mc_dropout_results(r_test_interp, r_test_extrap,
                              B_true_test_interp, B_true_test_extrap,
                              mean_interp, mean_extrap,
                              std_interp, std_extrap,
                              results_interp, results_extrap):
    """
    Create comprehensive 6-plot visualization for MC Dropout results.

    Args:
        r_test_interp: Distance values for interpolation test
        r_test_extrap: Distance values for extrapolation test
        B_true_test_interp: True field values (interpolation)
        B_true_test_extrap: True field values (extrapolation)
        mean_interp: Predicted means (interpolation)
        mean_extrap: Predicted means (extrapolation)
        std_interp: Predicted uncertainties (interpolation)
        std_extrap: Predicted uncertainties (extrapolation)
        results_interp: Full results dict with percentiles (interpolation)
        results_extrap: Full results dict with percentiles (extrapolation)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Monte Carlo Dropout: EM Field Prediction with Uncertainty',
                 fontsize=16, fontweight='bold')

    # Plot 1: Interpolation predictions with confidence intervals
    idx_sorted = torch.argsort(r_test_interp)[:30]
    axes[0, 0].plot(r_test_interp[idx_sorted].numpy(), B_true_test_interp[idx_sorted].numpy(),
                   'k-o', label='True Field', markersize=4, linewidth=2)
    axes[0, 0].plot(r_test_interp[idx_sorted].numpy(), mean_interp[idx_sorted].numpy(),
                   'b-s', label='MC Prediction', markersize=4)
    axes[0, 0].fill_between(r_test_interp[idx_sorted].numpy(),
                            results_interp['percentile_5'][idx_sorted].numpy().flatten(),
                            results_interp['percentile_95'][idx_sorted].numpy().flatten(),
                            alpha=0.3, color='blue', label='90% CI')
    axes[0, 0].set_xlabel('Distance r (m)', fontsize=10)
    axes[0, 0].set_ylabel('B-field (normalized)', fontsize=10)
    axes[0, 0].set_title('Interpolation: Predictions Within Training Range', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Extrapolation predictions
    idx_sorted_extrap = torch.argsort(r_test_extrap)[:30]
    axes[0, 1].plot(r_test_extrap[idx_sorted_extrap].numpy(),
                   B_true_test_extrap[idx_sorted_extrap].numpy(),
                   'k-o', label='True Field', markersize=4, linewidth=2)
    axes[0, 1].plot(r_test_extrap[idx_sorted_extrap].numpy(),
                   mean_extrap[idx_sorted_extrap].numpy(),
                   'r-s', label='MC Prediction', markersize=4)
    axes[0, 1].fill_between(r_test_extrap[idx_sorted_extrap].numpy(),
                            results_extrap['percentile_5'][idx_sorted_extrap].numpy().flatten(),
                            results_extrap['percentile_95'][idx_sorted_extrap].numpy().flatten(),
                            alpha=0.3, color='red', label='90% CI (Wider!)')
    axes[0, 1].axvline(x=0.15, color='green', linestyle='--', linewidth=2,
                      label='Training boundary')
    axes[0, 1].set_xlabel('Distance r (m)', fontsize=10)
    axes[0, 1].set_ylabel('B-field (normalized)', fontsize=10)
    axes[0, 1].set_title('Extrapolation: Predictions BEYOND Training Range', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Uncertainty comparison
    uncertainty_data = [std_interp.numpy().flatten(), std_extrap.numpy().flatten()]
    bp = axes[0, 2].boxplot(uncertainty_data, labels=['Interpolation', 'Extrapolation'],
                            patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[0, 2].set_ylabel('Epistemic Uncertainty (σ)', fontsize=10)
    axes[0, 2].set_title('Uncertainty Increases for Extrapolation', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    axes[0, 2].text(1, std_interp.mean().item(), f'μ={std_interp.mean().item():.4f}',
                   ha='center', fontsize=9, fontweight='bold')
    axes[0, 2].text(2, std_extrap.mean().item(), f'μ={std_extrap.mean().item():.4f}',
                   ha='center', fontsize=9, fontweight='bold')

    # Plot 4: Uncertainty vs Error (Interpolation)
    errors_interp = torch.abs(mean_interp - B_true_test_interp)
    axes[1, 0].scatter(std_interp.numpy().flatten(), errors_interp.numpy().flatten(),
                      alpha=0.6, c='blue')
    axes[1, 0].set_xlabel('Predicted Uncertainty (σ)', fontsize=10)
    axes[1, 0].set_ylabel('Absolute Error |ŷ - y|', fontsize=10)
    axes[1, 0].set_title('Interpolation: Uncertainty vs Error', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    corr_interp = np.corrcoef(std_interp.numpy().flatten(), errors_interp.numpy().flatten())[0, 1]
    axes[1, 0].text(0.05, 0.95, f'Correlation: {corr_interp:.3f}',
                   transform=axes[1, 0].transAxes, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 5: Uncertainty vs Error (Extrapolation)
    errors_extrap = torch.abs(mean_extrap - B_true_test_extrap)
    axes[1, 1].scatter(std_extrap.numpy().flatten(), errors_extrap.numpy().flatten(),
                      alpha=0.6, c='red')
    axes[1, 1].set_xlabel('Predicted Uncertainty (σ)', fontsize=10)
    axes[1, 1].set_ylabel('Absolute Error |ŷ - y|', fontsize=10)
    axes[1, 1].set_title('Extrapolation: Uncertainty vs Error', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    corr_extrap = np.corrcoef(std_extrap.numpy().flatten(), errors_extrap.numpy().flatten())[0, 1]
    axes[1, 1].text(0.05, 0.95, f'Correlation: {corr_extrap:.3f}',
                   transform=axes[1, 1].transAxes, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 6: Error comparison
    error_data = [errors_interp.numpy().flatten(), errors_extrap.numpy().flatten()]
    bp2 = axes[1, 2].boxplot(error_data, labels=['Interpolation', 'Extrapolation'],
                             patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightcoral')
    axes[1, 2].set_ylabel('Absolute Error', fontsize=10)
    axes[1, 2].set_title('Error Increases for Extrapolation', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def plot_ensemble_results(mean_pred, std_pred, true_values, ensemble_results):
    """
    Create 4-plot visualization for Deep Ensemble results.

    Args:
        mean_pred: Ensemble mean predictions
        std_pred: Ensemble uncertainty (std deviation)
        true_values: Ground truth values
        ensemble_results: Full results dict with 'all_predictions'
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Deep Ensemble: Uncertainty Quantification Results',
                 fontsize=16, fontweight='bold')

    # Plot 1: Ensemble mean with confidence intervals
    sample_idx = torch.arange(30)
    axes[0, 0].plot(sample_idx.numpy(), true_values[:30].numpy(), 'k-o',
                   label='True Values', markersize=4)
    axes[0, 0].plot(sample_idx.numpy(), mean_pred[:30].numpy(), 'b-s',
                   label='Ensemble Mean', markersize=4)
    axes[0, 0].fill_between(sample_idx.numpy(),
                            (mean_pred[:30] - 2*std_pred[:30]).numpy().flatten(),
                            (mean_pred[:30] + 2*std_pred[:30]).numpy().flatten(),
                            alpha=0.3, color='blue', label='95% CI')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Magnetic Field')
    axes[0, 0].set_title('Ensemble Prediction with Uncertainty')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Uncertainty vs Error
    errors = torch.abs(mean_pred - true_values)
    axes[0, 1].scatter(std_pred.numpy().flatten(), errors.numpy().flatten(), alpha=0.6)
    axes[0, 1].set_xlabel('Ensemble Uncertainty (σ)')
    axes[0, 1].set_ylabel('Absolute Error')
    axes[0, 1].set_title('Uncertainty vs Prediction Error')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Individual model predictions
    individual_predictions = ensemble_results['all_predictions']
    for i in range(min(3, individual_predictions.shape[0])):
        axes[1, 0].plot(sample_idx.numpy(), individual_predictions[i, :30].numpy(),
                       alpha=0.6, label=f'Model {i+1}')
    axes[1, 0].plot(sample_idx.numpy(), true_values[:30].numpy(), 'k-o',
                   label='True Values', markersize=4, linewidth=2)
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Magnetic Field')
    axes[1, 0].set_title('Individual Model Predictions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Uncertainty distribution
    axes[1, 1].hist(std_pred.numpy().flatten(), bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Ensemble Uncertainty (σ)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Uncertainty Distribution')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
