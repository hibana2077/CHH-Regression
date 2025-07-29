"""
MM-IRLS Optimization Algorithm for CHH Regression
Implements the Majorization-Minimization Iteratively Reweighted Least Squares algorithm
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
try:
    from .loss_functions import CHHLoss
except ImportError:
    from loss_functions import CHHLoss
import warnings


class CHHRegressor:
    """
    CHH Regressor using MM-IRLS algorithm
    
    Solves robust regression with Combined Huber-Correntropy Hybrid loss
    using Majorization-Minimization with Iteratively Reweighted Least Squares
    """
    
    def __init__(
        self,
        delta: Optional[float] = None,
        sigma: Optional[float] = None,
        beta: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        init_method: str = 'huber'
    ):
        """
        Initialize CHH Regressor
        
        Args:
            delta: Huber threshold (auto-estimated if None)
            sigma: Correntropy kernel width (auto-estimated if None)
            beta: Weight of correntropy component
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            fit_intercept: Whether to fit intercept term
            init_method: Initialization method ('ols', 'huber', 'zero')
        """
        self.delta = delta
        self.sigma = sigma
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.init_method = init_method
        
        # Will be set during fitting
        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []
        self.n_iter_ = 0
        self.chh_loss_ = None
    
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column to design matrix"""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def _initial_fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Get initial coefficient estimates using specified method"""
        if self.init_method == 'huber':
            try:
                from sklearn.linear_model import HuberRegressor
                huber = HuberRegressor(fit_intercept=False, max_iter=100)
                huber.fit(X, y)
                return huber.coef_
            except Exception:
                warnings.warn("Huber initialization failed, falling back to OLS")
                return self._ols_init(X, y)
        elif self.init_method == 'ols':
            return self._ols_init(X, y)
        elif self.init_method == 'zero':
            return np.zeros(X.shape[1])
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")
    
    def _ols_init(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """OLS initialization"""
        try:
            return np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            warnings.warn("OLS initialization failed, using zero initialization")
            return np.zeros(X.shape[1])
    
    def _compute_residuals(self, X: np.ndarray, y: np.ndarray, coef: np.ndarray) -> np.ndarray:
        """Compute residuals r = y - X @ coef"""
        return y - X @ coef
    
    def _wls_step(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Weighted Least Squares step
        Solve (X^T W X) beta = X^T W y
        """
        W_sqrt = np.sqrt(weights)
        X_weighted = X * W_sqrt[:, np.newaxis]
        y_weighted = y * W_sqrt
        
        try:
            return np.linalg.lstsq(X_weighted, y_weighted, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Fallback to regularized solution
            reg = 1e-8
            XTW = X.T * weights
            return np.linalg.solve(
                XTW @ X + reg * np.eye(X.shape[1]),
                XTW @ y
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CHHRegressor':
        """
        Fit CHH regression model using MM-IRLS algorithm
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        
        Returns:
            self: Fitted regressor
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        # Add intercept if needed
        if self.fit_intercept:
            X_design = self._add_intercept(X)
        else:
            X_design = X.copy()
        
        # Initial coefficient estimates
        coef = self._initial_fit(X_design, y)
        residuals = self._compute_residuals(X_design, y, coef)
        
        # Auto-estimate parameters if not provided
        try:
            from .loss_functions import get_default_parameters
        except ImportError:
            from loss_functions import get_default_parameters
        if self.delta is None or self.sigma is None:
            delta_auto, sigma_auto, _ = get_default_parameters(residuals)
            if self.delta is None:
                self.delta = delta_auto
            if self.sigma is None:
                self.sigma = sigma_auto
        
        # Initialize CHH loss function
        self.chh_loss_ = CHHLoss(self.delta, self.sigma, self.beta)
        
        # MM-IRLS iterations
        self.loss_history_ = []
        prev_loss = np.inf
        
        for iteration in range(self.max_iter):
            # Compute current loss
            current_loss = np.sum(self.chh_loss_.loss(residuals))
            self.loss_history_.append(current_loss)
            
            # Check convergence
            if iteration > 0:
                rel_change = abs(prev_loss - current_loss) / (abs(prev_loss) + 1e-12)
                if rel_change < self.tol:
                    break
            
            # Compute IRLS weights
            weights = self.chh_loss_.weights(residuals)
            
            # Ensure weights are positive and finite
            weights = np.maximum(weights, 1e-12)
            weights = np.where(np.isfinite(weights), weights, 1e-12)
            
            # WLS step
            coef = self._wls_step(X_design, y, weights)
            residuals = self._compute_residuals(X_design, y, coef)
            prev_loss = current_loss
        
        self.n_iter_ = iteration + 1
        
        # Store coefficients
        if self.fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coef
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using fitted model
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Predicted values (n_samples,)
        """
        if self.coef_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_
    
    def get_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get final IRLS weights for fitted model
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Returns:
            Final weights for each sample
        """
        if self.chh_loss_ is None:
            raise ValueError("Model must be fitted first")
        
        residuals = y - self.predict(X)
        return self.chh_loss_.weights(residuals)
    
    def score(self, X: np.ndarray, y: np.ndarray, metric: str = 'r2') -> float:
        """
        Compute score for fitted model
        
        Args:
            X: Feature matrix
            y: Target vector
            metric: Scoring metric ('r2', 'mae', 'rmse')
        
        Returns:
            Score value
        """
        y_pred = self.predict(X)
        
        if metric == 'r2':
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
        elif metric == 'mae':
            return np.mean(np.abs(y - y_pred))
        elif metric == 'rmse':
            return np.sqrt(np.mean((y - y_pred) ** 2))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator (sklearn compatibility)
        
        Args:
            deep: If True, return parameters for sub-estimators too
            
        Returns:
            Dictionary of parameter names and values
        """
        return {
            'delta': self.delta,
            'sigma': self.sigma,
            'beta': self.beta,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'fit_intercept': self.fit_intercept,
            'init_method': self.init_method
        }
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator (sklearn compatibility)
        
        Args:
            **params: Parameter names and values
            
        Returns:
            self
        """
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter {param} for estimator {type(self).__name__}")
        return self


def huber_regression(X: np.ndarray, y: np.ndarray, **kwargs) -> CHHRegressor:
    """Convenience function for pure Huber regression (beta=0)"""
    return CHHRegressor(beta=0.0, **kwargs).fit(X, y)


def welsch_regression(X: np.ndarray, y: np.ndarray, **kwargs) -> CHHRegressor:
    """Convenience function for pure Welsch/correntropy regression (large delta)"""
    kwargs.setdefault('delta', 1e6)  # Very large delta to approximate pure correntropy
    return CHHRegressor(**kwargs).fit(X, y)
