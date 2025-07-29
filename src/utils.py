"""
Utility functions for CHH-Regression
Helper functions for data processing, metrics, and common operations
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings


def mad_scale_estimate(residuals: np.ndarray) -> float:
    """
    Estimate scale using Median Absolute Deviation (MAD)
    
    Args:
        residuals: Array of residuals
    
    Returns:
        MAD-based scale estimate
    """
    median_residual = np.median(residuals)
    mad = np.median(np.abs(residuals - median_residual))
    # Consistency factor for normal distribution
    return 1.4826 * mad


def detect_outliers(
    residuals: np.ndarray,
    method: str = 'iqr',
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outliers in residuals
    
    Args:
        residuals: Array of residuals
        method: Detection method ('iqr', 'mad', 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean array indicating outliers
    """
    if method == 'iqr':
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (residuals < lower_bound) | (residuals > upper_bound)
    
    elif method == 'mad':
        median_res = np.median(residuals)
        mad = np.median(np.abs(residuals - median_res))
        modified_z_score = 0.6745 * (residuals - median_res) / mad
        return np.abs(modified_z_score) > threshold
    
    elif method == 'zscore':
        z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
        return z_scores > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def cross_validate_parameters(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, list],
    cv_folds: int = 5,
    scoring: str = 'neg_mean_absolute_error'
) -> Dict[str, Any]:
    """
    Cross-validate CHH parameters
    
    Args:
        X: Feature matrix
        y: Target vector
        param_grid: Dictionary of parameters to test
        cv_folds: Number of CV folds
        scoring: Scoring metric
    
    Returns:
        Best parameters and scores
    """
    from sklearn.model_selection import cross_val_score
    try:
        from .optimizer import CHHRegressor
    except ImportError:
        from optimizer import CHHRegressor
    from itertools import product
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    best_score = -np.inf
    best_params = None
    all_results = []
    
    for param_combo in product(*param_values):
        params = dict(zip(param_names, param_combo))
        
        try:
            estimator = CHHRegressor(**params, max_iter=50)
            scores = cross_val_score(estimator, X, y, cv=cv_folds, scoring=scoring)
            mean_score = np.mean(scores)
            
            all_results.append({
                'params': params.copy(),
                'mean_score': mean_score,
                'std_score': np.std(scores)
            })
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
                
        except Exception as e:
            warnings.warn(f"Failed for params {params}: {e}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': all_results
    }


def compute_influence_diagnostics(
    X: np.ndarray,
    y: np.ndarray,
    fitted_model: Any
) -> Dict[str, np.ndarray]:
    """
    Compute influence diagnostics for fitted model
    
    Args:
        X: Feature matrix
        y: Target vector
        fitted_model: Fitted CHH regressor
    
    Returns:
        Dictionary of influence diagnostics
    """
    residuals = y - fitted_model.predict(X)
    weights = fitted_model.get_weights(X, y)
    
    # Standardized residuals
    scale = mad_scale_estimate(residuals)
    std_residuals = residuals / scale
    
    # Leverage (hat values) - approximate for robust regression
    if hasattr(fitted_model, 'fit_intercept') and fitted_model.fit_intercept:
        X_design = np.column_stack([np.ones(X.shape[0]), X])
    else:
        X_design = X
    
    try:
        # Weighted hat matrix diagonal
        W = np.diag(weights)
        H = X_design @ np.linalg.inv(X_design.T @ W @ X_design) @ X_design.T @ W
        leverage = np.diag(H)
    except:
        # Fallback: uniform leverage
        leverage = np.full(len(y), X.shape[1] / len(y))
    
    # Cook's distance approximation
    p = X.shape[1]
    cooks_d = (std_residuals**2 / p) * (leverage / (1 - leverage)**2)
    
    return {
        'residuals': residuals,
        'std_residuals': std_residuals,
        'weights': weights,
        'leverage': leverage,
        'cooks_distance': cooks_d
    }


def bootstrap_confidence_intervals(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Any,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Compute bootstrap confidence intervals for coefficients
    
    Args:
        X: Feature matrix
        y: Target vector
        estimator: Unfitted estimator
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        random_state: Random seed
    
    Returns:
        Dictionary with coefficient CIs
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(y)
    bootstrap_coefs = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        try:
            # Fit estimator
            estimator_copy = type(estimator)(**estimator.get_params())
            estimator_copy.fit(X_boot, y_boot)
            
            # Store coefficients
            coefs = np.concatenate([[estimator_copy.intercept_], estimator_copy.coef_])
            bootstrap_coefs.append(coefs)
            
        except Exception as e:
            warnings.warn(f"Bootstrap iteration failed: {e}")
    
    if not bootstrap_coefs:
        raise RuntimeError("All bootstrap iterations failed")
    
    bootstrap_coefs = np.array(bootstrap_coefs)
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    ci_lower = np.percentile(bootstrap_coefs, lower_percentile, axis=0)
    ci_upper = np.percentile(bootstrap_coefs, upper_percentile, axis=0)
    
    return {
        'coefficients_mean': np.mean(bootstrap_coefs, axis=0),
        'coefficients_std': np.std(bootstrap_coefs, axis=0),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_samples': bootstrap_coefs
    }


def generate_synthetic_data(
    n_samples: int = 200,
    n_features: int = 5,
    noise_std: float = 1.0,
    outlier_fraction: float = 0.1,
    outlier_magnitude: float = 5.0,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data with outliers
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise_std: Standard deviation of noise
        outlier_fraction: Fraction of outliers
        outlier_magnitude: Magnitude of outliers
        random_state: Random seed
    
    Returns:
        Tuple of (X, y, true_coefficients)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # True coefficients
    true_coef = np.random.randn(n_features)
    
    # Generate clean target
    y_clean = X @ true_coef
    
    # Add normal noise
    noise = np.random.normal(0, noise_std, n_samples)
    y = y_clean + noise
    
    # Add outliers
    n_outliers = int(n_samples * outlier_fraction)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    outlier_noise = np.random.choice([-1, 1], n_outliers) * outlier_magnitude * noise_std
    y[outlier_indices] += outlier_noise
    
    return X, y, true_coef


def validate_input_data(X: np.ndarray, y: np.ndarray) -> None:
    """
    Validate input data for regression
    
    Args:
        X: Feature matrix
        y: Target vector
    
    Raises:
        ValueError: If data is invalid
    """
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X and y must be numpy arrays")
    
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    
    if y.ndim != 1:
        raise ValueError("y must be 1-dimensional")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    
    if X.shape[0] < X.shape[1]:
        warnings.warn("Number of samples is less than number of features")
    
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Data contains NaN values")
    
    if np.any(np.isinf(X)) or np.any(np.isinf(y)):
        raise ValueError("Data contains infinite values")


def add_outliers(
    y: np.ndarray,
    contamination_rate: float = 0.1,
    outlier_magnitude: float = 5.0,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Add outliers to target array
    
    Args:
        y: Original target array
        contamination_rate: Fraction of outliers to add
        outlier_magnitude: Magnitude of outliers (in standard deviations)
        random_state: Random seed for reproducibility
    
    Returns:
        Target array with added outliers
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    y_contaminated = y.copy()
    n_samples = len(y)
    n_outliers = int(n_samples * contamination_rate)
    
    if n_outliers > 0:
        # Select random indices for outliers
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        
        # Estimate scale
        scale = mad_scale_estimate(y - np.median(y))
        if scale == 0:
            scale = np.std(y)
        
        # Add outliers with random signs
        outlier_signs = np.random.choice([-1, 1], n_outliers)
        outlier_values = outlier_signs * outlier_magnitude * scale
        
        y_contaminated[outlier_indices] += outlier_values
    
    return y_contaminated


def compute_robust_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute robust regression metrics
    
    Args:
        y_true: True target values
        y_pred: Predicted values
    
    Returns:
        Dictionary of robust metrics
    """
    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)
    
    metrics = {
        'mae': np.mean(abs_residuals),
        'rmse': np.sqrt(np.mean(residuals**2)),
        'median_ae': np.median(abs_residuals),
        'mad_residuals': mad_scale_estimate(residuals),
        'q75_ae': np.percentile(abs_residuals, 75),
        'q90_ae': np.percentile(abs_residuals, 90),
        'q95_ae': np.percentile(abs_residuals, 95),
        'max_ae': np.max(abs_residuals),
        'huber_loss': np.mean(np.where(abs_residuals <= 1.345, 
                                     0.5 * residuals**2,
                                     1.345 * abs_residuals - 0.5 * 1.345**2))
    }
    
    # R-squared variants
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    metrics['r2'] = 1 - (ss_res / ss_tot)
    
    # Robust R-squared using median
    median_y = np.median(y_true)
    mad_y = np.median(np.abs(y_true - median_y))
    mad_res = np.median(abs_residuals)
    metrics['robust_r2'] = 1 - (mad_res / mad_y)**2
    
    return metrics
