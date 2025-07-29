"""
Evaluation Module for CHH-Regression
Implements comprehensive evaluation metrics and robustness tests
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics
    
    Args:
        y_true: True target values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    residuals = y_true - y_pred
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'median_abs_residual': np.median(np.abs(residuals)),
        'q90_abs_residual': np.percentile(np.abs(residuals), 90),
        'q95_abs_residual': np.percentile(np.abs(residuals), 95),
        'max_abs_residual': np.max(np.abs(residuals))
    }
    
    return metrics


def robust_cross_validation(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    contamination_rates: List[float] = [0.0, 0.05, 0.1, 0.2],
    outlier_magnitude: float = 8.0,
    random_state: Optional[int] = None
) -> Dict[str, Dict[str, List[float]]]:
    """
    Perform cross-validation with various contamination levels
    
    Args:
        estimator: Regression estimator
        X: Feature matrix
        y: Target vector
        cv: Number of CV folds
        contamination_rates: List of contamination rates to test
        outlier_magnitude: Magnitude of artificial outliers
        random_state: Random seed
    
    Returns:
        Dictionary of results for each contamination rate
    """
    from sklearn.model_selection import KFold
    try:
        from .data_loader import add_outliers
    except ImportError:
        from data_loader import add_outliers
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = {}
    
    for cont_rate in contamination_rates:
        fold_results = {'mae': [], 'rmse': [], 'r2': []}
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Add contamination to training data
            if cont_rate > 0:
                y_train_cont, _ = add_outliers(
                    y_train, cont_rate, outlier_magnitude, random_state
                )
            else:
                y_train_cont = y_train
            
            # Fit and predict
            estimator.fit(X_train, y_train_cont)
            y_pred = estimator.predict(X_test)
            
            # Compute metrics on clean test data
            metrics = compute_metrics(y_test, y_pred)
            fold_results['mae'].append(metrics['mae'])
            fold_results['rmse'].append(metrics['rmse'])
            fold_results['r2'].append(metrics['r2'])
        
        results[f'contamination_{cont_rate:.2f}'] = fold_results
    
    return results


class RegressionBenchmark:
    """
    Comprehensive benchmark for robust regression methods
    """
    
    def __init__(self, random_state: Optional[int] = 42):
        self.random_state = random_state
        self.results = {}
    
    def create_estimators(self) -> Dict[str, Any]:
        """Create dictionary of estimators to compare"""
        try:
            from .optimizer import CHHRegressor
            from .mcc_regressor import MCCRegressor
        except ImportError:
            from optimizer import CHHRegressor
            from mcc_regressor import MCCRegressor
        
        estimators = {
            'OLS': LinearRegression(),
            'Huber': HuberRegressor(epsilon=1.35, max_iter=100),
            'MCC_sigma0.6': MCCRegressor(sigma=0.6, max_iter=100),
            'MCC_sigma0.8': MCCRegressor(sigma=0.8, max_iter=100),  
            'MCC_sigma1.0': MCCRegressor(sigma=1.0, max_iter=100),
            'CHH_beta8.0_sigma0.2': CHHRegressor(beta=8.0, max_iter=100, sigma=0.2),
            'CHH_beta8.0_sigma0.4': CHHRegressor(beta=8.0, max_iter=100, sigma=0.4),
            'CHH_beta8.0_sigma1.0': CHHRegressor(beta=8.0, max_iter=100, sigma=1.0),
            'CHH_beta12.0_sigma0.2': CHHRegressor(beta=12.0, max_iter=100, sigma=0.2),
            'CHH_beta12.0_sigma0.4': CHHRegressor(beta=12.0, max_iter=100, sigma=0.4),
            'CHH_beta12.0_sigma1.0': CHHRegressor(beta=12.0, max_iter=100, sigma=1.0),
        }
        
        return estimators
    
    def run_benchmark(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        contamination_rates: List[float] = [0.0, 0.05, 0.1, 0.2]
    ) -> Dict[str, Any]:
        """
        Run complete benchmark on dataset
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data for testing
            contamination_rates: Contamination rates to test
        
        Returns:
            Comprehensive benchmark results
        """
        estimators = self.create_estimators()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        results = {}
        
        for name, estimator in estimators.items():
            print(f"Evaluating {name}...")
            
            # Cross-validation with contamination
            cv_results = robust_cross_validation(
                estimator, X_train, y_train,
                contamination_rates=contamination_rates,
                random_state=self.random_state
            )
            
            # Final test set evaluation
            test_results = {}
            for cont_rate in contamination_rates:
                # Train on contaminated data
                if cont_rate > 0:
                    try:
                        from .data_loader import add_outliers
                    except ImportError:
                        from data_loader import add_outliers
                    y_train_cont, _ = add_outliers(
                        y_train, cont_rate, 8.0, self.random_state
                    )
                else:
                    y_train_cont = y_train
                
                # Fit and predict
                try:
                    estimator.fit(X_train, y_train_cont)
                    y_pred = estimator.predict(X_test)
                    test_metrics = compute_metrics(y_test, y_pred)
                    test_results[f'contamination_{cont_rate:.2f}'] = test_metrics
                except Exception as e:
                    warnings.warn(f"Error with {name} at contamination {cont_rate}: {e}")
                    test_results[f'contamination_{cont_rate:.2f}'] = {
                        'mae': np.inf, 'rmse': np.inf, 'r2': -np.inf
                    }
            
            results[name] = {
                'cv_results': cv_results,
                'test_results': test_results
            }
        
        return results
    
    def print_summary(self, results: Dict[str, Any], metric: str = 'rmse') -> None:
        """
        Print summary table of results
        
        Args:
            results: Benchmark results
            metric: Metric to display ('mae', 'rmse', 'r2')
        """
        print(f"\n=== {metric.upper()} Results Summary ===")
        
        # Get contamination rates
        first_method = list(results.keys())[0]
        cont_rates = list(results[first_method]['test_results'].keys())
        
        # Header
        header = f"{'Method':<15}"
        for cont_rate in cont_rates:
            rate_str = cont_rate.replace('contamination_', '')
            header += f"{rate_str:>10}"
        print(header)
        print("-" * len(header))
        
        # Results for each method
        for method_name, method_results in results.items():
            row = f"{method_name:<15}"
            for cont_rate in cont_rates:
                value = method_results['test_results'][cont_rate][metric]
                if np.isfinite(value):
                    row += f"{value:>10.4f}"
                else:
                    row += f"{'FAIL':>10}"
            print(row)


def parameter_sensitivity_analysis(
    X: np.ndarray,
    y: np.ndarray,
    delta_range: List[float] = [1.345, 1.6, 2.0],
    sigma_range: List[float] = [0.5, 0.7, 1.0, 1.2],
    beta_range: List[float] = [0.02, 0.05, 0.1, 0.2],
    cv_folds: int = 3,
    scoring: str = 'neg_mean_absolute_error'
) -> Dict[str, Any]:
    """
    Analyze sensitivity to CHH parameters
    
    Args:
        X: Feature matrix
        y: Target vector
        delta_range: Range of delta values to test (as multiples of sigma_hat)
        sigma_range: Range of sigma values to test (as multiples of sigma_hat)
        beta_range: Range of beta values to test
        cv_folds: Number of CV folds
        scoring: CV scoring metric ('neg_mean_absolute_error' or 'neg_mean_squared_error')
    
    Returns:
        Parameter sensitivity results
    """
    try:
        from .optimizer import CHHRegressor
        from .loss_functions import get_default_parameters
    except ImportError:
        from optimizer import CHHRegressor
        from loss_functions import get_default_parameters
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    
    # Estimate sigma_hat from OLS residuals for parameter scaling
    ols = LinearRegression()
    ols.fit(X, y)
    ols_residuals = y - ols.predict(X)
    _, sigma_hat, _ = get_default_parameters(ols_residuals)
    
    results = []
    print(f"Using sigma_hat = {sigma_hat:.4f} for parameter scaling")
    print(f"Testing {len(delta_range) * len(sigma_range) * len(beta_range)} parameter combinations...")
    
    total_combinations = len(delta_range) * len(sigma_range) * len(beta_range)
    failed_count = 0
    
    for delta_scale in delta_range:
        for sigma_scale in sigma_range:
            for beta in beta_range:
                try:
                    # Scale parameters by sigma_hat
                    delta_actual = delta_scale * sigma_hat
                    sigma_actual = sigma_scale * sigma_hat
                    
                    print(f"Testing: δ={delta_actual:.4f}, σ={sigma_actual:.4f}, β={beta}")
                    
                    estimator = CHHRegressor(
                        delta=delta_actual, sigma=sigma_actual, beta=beta, 
                        max_iter=50, init_method='huber'
                    )
                    
                    # Use MAE as default scoring (better for robustness evaluation)
                    scores = cross_val_score(
                        estimator, X, y, 
                        cv=cv_folds, 
                        scoring=scoring
                    )
                    
                    results.append({
                        'delta_scale': delta_scale,
                        'sigma_scale': sigma_scale,
                        'delta_actual': delta_actual,
                        'sigma_actual': sigma_actual,
                        'beta': beta,
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores),
                        'sigma_hat': sigma_hat
                    })
                    print(f"  ✓ Success: MAE = {-np.mean(scores):.4f}")
                    
                except Exception as e:
                    failed_count += 1
                    print(f"  ✗ Failed: {e}")
                    warnings.warn(f"Failed for delta_scale={delta_scale}, sigma_scale={sigma_scale}, beta={beta}: {e}")
    
    print(f"Completed: {len(results)}/{total_combinations} successful, {failed_count} failed")
    
    return results


def compare_regressors(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    contamination_rates: List[float] = [0.0, 0.05, 0.1, 0.2],
    outlier_magnitude: float = 8.0,
    random_state: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple robust regression methods
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Test set proportion
        contamination_rates: List of contamination rates to test
        outlier_magnitude: Magnitude of outliers (in MAD units)
        random_state: Random seed
        
    Returns:
        Dictionary with results for each method and contamination rate
    """
    try:
        from .optimizer import CHHRegressor
        from .mcc_regressor import MCCRegressor
        from .utils import add_outliers, estimate_scale
    except ImportError:
        from optimizer import CHHRegressor
        from mcc_regressor import MCCRegressor
        from utils import add_outliers, estimate_scale
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Estimate scale for parameter setting
    initial_residuals = y_train - np.mean(y_train)
    scale = estimate_scale(initial_residuals, method='mad')
    
    # Create regressors
    regressors = {
        'OLS': LinearRegression(),
        'Huber': HuberRegressor(epsilon=1.35, max_iter=100),
        'MCC': MCCRegressor(sigma=scale, max_iter=100),
        'CHH': CHHRegressor(
            delta=1.345*scale, 
            sigma=scale, 
            beta=1.0, 
            max_iter=100
        )
    }
    
    results = {}
    
    for name, regressor in regressors.items():
        results[name] = {}
        
        for contamination in contamination_rates:
            # Add outliers to training data
            y_train_cont = add_outliers(
                y_train.copy(), 
                contamination_rate=contamination,
                outlier_magnitude=outlier_magnitude * scale,
                random_state=random_state
            )
            
            try:
                # Fit regressor
                regressor.fit(X_train, y_train_cont)
                
                # Predict on clean test set
                y_pred = regressor.predict(X_test)
                
                # Compute metrics
                metrics = compute_metrics(y_test, y_pred)
                results[name][f'contamination_{contamination}'] = metrics
                
            except Exception as e:
                warnings.warn(f"Failed to fit {name} with contamination {contamination}: {e}")
                results[name][f'contamination_{contamination}'] = {
                    'mae': np.inf, 'rmse': np.inf, 'r2': -np.inf
                }
    
    return results


def convergence_analysis(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 100
) -> Dict[str, Any]:
    """
    Analyze convergence properties of CHH algorithm
    
    Args:
        X: Feature matrix
        y: Target vector
        max_iter: Maximum iterations to analyze
    
    Returns:
        Convergence analysis results
    """
    try:
        from .optimizer import CHHRegressor
    except ImportError:
        from optimizer import CHHRegressor
    
    estimator = CHHRegressor(max_iter=max_iter, tol=0)  # Disable early stopping
    estimator.fit(X, y)
    
    return {
        'loss_history': estimator.loss_history_,
        'n_iterations': estimator.n_iter_,
        'final_loss': estimator.loss_history_[-1] if estimator.loss_history_ else None,
        'converged': estimator.n_iter_ < max_iter
    }
