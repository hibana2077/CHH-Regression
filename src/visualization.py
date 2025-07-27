"""
Visualization Module for CHH-Regression
Provides plotting functions for analysis and results visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
import seaborn as sns
from pathlib import Path


# Set matplotlib backend and style
plt.style.use('default')
sns.set_palette("husl")


def plot_loss_functions(
    residual_range: Tuple[float, float] = (-5, 5),
    delta: float = 1.345,
    sigma: float = 1.0,
    beta: float = 1.0,
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison of different loss functions
    
    Args:
        residual_range: Range of residuals to plot
        delta: Huber threshold
        sigma: Correntropy kernel width
        beta: CHH beta parameter
        save_path: Path to save figure
    """
    try:
        from .loss_functions import CHHLoss
    except ImportError:
        from loss_functions import CHHLoss
    
    r = np.linspace(residual_range[0], residual_range[1], 1000)
    
    # Create loss functions
    chh = CHHLoss(delta, sigma, beta)
    huber = CHHLoss(delta, sigma, 0.0)  # beta=0 for pure Huber
    
    # Compute losses
    mse_loss = 0.5 * r**2
    huber_loss = huber.loss(r)
    chh_loss = chh.loss(r)
    corr_loss = 1 - np.exp(-r**2 / (2 * sigma**2))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss functions
    ax1.plot(r, mse_loss, 'b--', label='MSE', linewidth=2)
    ax1.plot(r, huber_loss, 'g-', label='Huber', linewidth=2)
    ax1.plot(r, corr_loss, 'r:', label='Correntropy', linewidth=2)
    ax1.plot(r, chh_loss, 'orange', label=f'CHH (β={beta})', linewidth=3)
    ax1.set_xlabel('Residual')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Functions Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Influence functions (derivatives)
    mse_psi = r
    huber_psi = huber.psi(r)
    chh_psi = chh.psi(r)
    corr_psi = (r / sigma**2) * np.exp(-r**2 / (2 * sigma**2))
    
    ax2.plot(r, mse_psi, 'b--', label='MSE', linewidth=2)
    ax2.plot(r, huber_psi, 'g-', label='Huber', linewidth=2)
    ax2.plot(r, corr_psi, 'r:', label='Correntropy', linewidth=2)
    ax2.plot(r, chh_psi, 'orange', label=f'CHH (β={beta})', linewidth=3)
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Influence Function ψ(r)')
    ax2.set_title('Influence Functions Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_convergence(
    loss_history: List[float],
    title: str = "CHH-MM-IRLS Convergence",
    save_path: Optional[str] = None
) -> None:
    """
    Plot convergence history
    
    Args:
        loss_history: List of loss values per iteration
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('CHH Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add convergence info
    if len(loss_history) > 1:
        final_loss = loss_history[-1]
        plt.axhline(y=final_loss, color='r', linestyle='--', alpha=0.7,
                   label=f'Final Loss: {final_loss:.4f}')
        plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_residuals_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
    method_name: str = "CHH",
    save_path: Optional[str] = None
) -> None:
    """
    Plot residuals analysis including fitted vs actual and residual plots
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        weights: Sample weights (optional)
        method_name: Name of the method
        save_path: Path to save figure
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Fitted vs Actual
    ax1 = axes[0, 0]
    ax1.scatter(y_true, y_pred, alpha=0.6)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'{method_name}: Predicted vs True')
    ax1.grid(True, alpha=0.3)
    
    # Residuals vs Fitted
    ax2 = axes[0, 1]
    ax2.scatter(y_pred, residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Fitted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'{method_name}: Residuals vs Fitted')
    ax2.grid(True, alpha=0.3)
    
    # Residuals histogram
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Density')
    ax3.set_title(f'{method_name}: Residuals Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Weights plot (if available)
    ax4 = axes[1, 1]
    if weights is not None:
        ax4.scatter(np.abs(residuals), weights, alpha=0.6)
        ax4.set_xlabel('|Residuals|')
        ax4.set_ylabel('Weights')
        ax4.set_title(f'{method_name}: Weights vs |Residuals|')
    else:
        ax4.text(0.5, 0.5, 'Weights not available', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Weights Analysis')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_benchmark_results(
    results: Dict[str, Any],
    metric: str = 'rmse',
    save_path: Optional[str] = None
) -> None:
    """
    Plot benchmark results across different contamination rates
    
    Args:
        results: Benchmark results from evaluation module
        metric: Metric to plot ('mae', 'rmse', 'r2')
        save_path: Path to save figure
    """
    # Extract data
    methods = list(results.keys())
    first_method = methods[0]
    cont_rates = list(results[first_method]['test_results'].keys())
    
    # Prepare data for plotting
    plot_data = []
    for method in methods:
        for cont_rate in cont_rates:
            rate_val = float(cont_rate.replace('contamination_', ''))
            metric_val = results[method]['test_results'][cont_rate][metric]
            if np.isfinite(metric_val):
                plot_data.append({
                    'Method': method,
                    'Contamination Rate': rate_val,
                    metric.upper(): metric_val
                })
    
    # Create DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame(plot_data)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    for method in methods:
        method_data = df[df['Method'] == method]
        if not method_data.empty:
            plt.plot(method_data['Contamination Rate'], method_data[metric.upper()],
                    marker='o', linewidth=2, markersize=6, label=method)
    
    plt.xlabel('Contamination Rate')
    plt.ylabel(metric.upper())
    plt.title(f'Robustness Comparison: {metric.upper()} vs Contamination Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_parameter_sensitivity(
    sensitivity_results: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot parameter sensitivity analysis results
    
    Args:
        sensitivity_results: Results from parameter_sensitivity_analysis
        save_path: Path to save figure
    """
    import pandas as pd
    
    df = pd.DataFrame(sensitivity_results)
    
    # Create subplots for each parameter
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Beta sensitivity
    beta_df = df.groupby('beta')['mean_score'].mean().reset_index()
    axes[0].plot(beta_df['beta'], -beta_df['mean_score'], 'o-', linewidth=2, markersize=6)
    axes[0].set_xlabel('β (Beta)')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('Sensitivity to β Parameter')
    axes[0].grid(True, alpha=0.3)
    
    # Delta sensitivity
    delta_df = df.groupby('delta')['mean_score'].mean().reset_index()
    axes[1].plot(delta_df['delta'], -delta_df['mean_score'], 'o-', linewidth=2, markersize=6)
    axes[1].set_xlabel('δ (Delta)')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Sensitivity to δ Parameter')
    axes[1].grid(True, alpha=0.3)
    
    # Sigma sensitivity
    sigma_df = df.groupby('sigma')['mean_score'].mean().reset_index()
    axes[2].plot(sigma_df['sigma'], -sigma_df['mean_score'], 'o-', linewidth=2, markersize=6)
    axes[2].set_xlabel('σ (Sigma)')
    axes[2].set_ylabel('MAE')
    axes[2].set_title('Sensitivity to σ Parameter')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_results_report(
    results: Dict[str, Any],
    dataset_name: str,
    output_dir: str = "results"
) -> None:
    """
    Create comprehensive visual report of results
    
    Args:
        results: Benchmark results
        dataset_name: Name of dataset
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Plot RMSE results
    plot_benchmark_results(
        results, 'rmse',
        save_path=output_path / f"{dataset_name}_rmse_comparison.png"
    )
    
    # Plot MAE results
    plot_benchmark_results(
        results, 'mae',
        save_path=output_path / f"{dataset_name}_mae_comparison.png"
    )
    
    # Plot R² results
    plot_benchmark_results(
        results, 'r2',
        save_path=output_path / f"{dataset_name}_r2_comparison.png"
    )
    
    print(f"Results plots saved to {output_path}")


def plot_weights_distribution(
    residuals: np.ndarray,
    weights: np.ndarray,
    method_name: str = "CHH",
    save_path: Optional[str] = None
) -> None:
    """
    Plot the distribution of weights vs residuals
    
    Args:
        residuals: Residual values
        weights: Corresponding weights
        method_name: Name of the method
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    # Sort by absolute residuals for cleaner plot
    abs_residuals = np.abs(residuals)
    sort_idx = np.argsort(abs_residuals)
    
    plt.scatter(abs_residuals[sort_idx], weights[sort_idx], alpha=0.6, s=20)
    plt.xlabel('|Residuals|')
    plt.ylabel('Weights')
    plt.title(f'{method_name}: Weight Assignment vs Absolute Residuals')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(abs_residuals, weights, 3)
    p = np.poly1d(z)
    x_smooth = np.linspace(abs_residuals.min(), abs_residuals.max(), 100)
    plt.plot(x_smooth, p(x_smooth), 'r-', linewidth=2, alpha=0.8, label='Trend')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
