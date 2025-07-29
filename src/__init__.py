# CHH-Regression Package

from .loss_functions import CHHLoss, WelschLoss, estimate_scale, get_default_parameters
from .optimizer import CHHRegressor
from .mcc_regressor import MCCRegressor, auto_mcc_regressor
from .data_loader import DataLoader, load_airfoil_data, load_yacht_data, load_california_housing
from .evaluation import compare_regressors, parameter_sensitivity_analysis, convergence_analysis, compute_metrics
from .visualization import plot_loss_functions, plot_convergence, plot_residuals_analysis, plot_benchmark_results
from .utils import add_outliers, compute_robust_metrics, mad_scale_estimate, detect_outliers

__all__ = [
    'CHHLoss',
    'WelschLoss', 
    'CHHRegressor',
    'MCCRegressor',
    'auto_mcc_regressor',
    'load_dataset',
    'evaluate_regressor',
    'compare_regressors',
    'plot_results',
    'plot_loss_comparison',
    'add_outliers',
    'compute_metrics',
    'compute_robust_metrics',
    'estimate_scale',
    'get_default_parameters'
]
